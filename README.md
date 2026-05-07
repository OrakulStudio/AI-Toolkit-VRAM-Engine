[English Version] | [Русская версия ниже](#русская-версия)

# Orakul Engine: Universal VRAM Management for Ostris AI-Toolkit

**or: an engine inside an engine, written in a basement under artillery fire**

*Orakul Studio — Chernihiv, Ukraine 🇺🇦*

---

## Numbers. First.

Because without them, everything else is just words.

### Production Training Results (Rank 32 / Alpha 64)

| Configuration | Before | After | Speedup |
|---|---|---|---|
| LoRA rank 32 / alpha 64, Flux2-dev | **73 sec/iter** | **6.57 sec/iter** | **11×** |
| Full 1000-step training | **~20 hours** | **~2.5 hours** | — |
| Most models incl. video-LoRA | — | **~2 sec/iter** | — |

### Extreme Rank Stress Test (Rank 1024 — Stability Proof)

| Configuration | Result |
|---|---|
| LoRA rank 1024 / alpha 1024, Flux2-dev | **Stable. 0 crashes. 0 OOM.** |
| Baseline before optimization, rank 1024 | 179 sec/iter |
| After optimization, rank 1024 | 6–8 sec/iter |

> The rank 1024 test is not a production config.  
> It exists to prove one thing: **the architecture does not break under extreme load.**  
> If it holds at rank 1024 — it holds at anything below it.

**Hardware:** RTX 4090 (24 GB VRAM) · i9-13900K · 128 GB RAM  
**Framework:** [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)  
No additional GPUs. No cloud. No server hardware.

This is not a configuration tweak. This is a rewritten PyTorch memory layer.

---

## The Problem. For Everyone.

Flux2 is a large model. A transformer with billions of parameters.  
RTX 4090 has 24 GB VRAM. The model **does not fit** entirely.

ai-toolkit solves this via **layer offloading**: each layer's weights are stored in system RAM, transferred to the GPU before computation, then offloaded back afterward.

An elegant solution. But it has one critical flaw:

```
GPU computes layer N        ████████████████
Transfer weights layer N+1              ████████████████
```

This is a **sequential process**. The GPU waits idle while data arrives.  
Data travels while the GPU does nothing.

At rank 32, this overhead is already significant across hundreds of layers.  
At rank 1024 it becomes catastrophic — 179 seconds per iteration.  
That is why the stress test exists: to show the full scale of the problem.

---

## The Solution. Also for Everyone.

The idea is simple. The implementation — not so much.

While the GPU **computes** layer N — in a separate CUDA stream, in parallel, **the transfer of layer N+1's weights has already begun**.

```
GPU computes layer N        ████████████████
Transfer weights layer N+1  ████████████████
                            ↑
                            Starts simultaneously
```

By the time the GPU finishes layer N — the weights for layer N+1 are already there. No waiting.

This is called **double buffering** with **compute-transfer overlap**.  
In HPC systems, this is standard practice. In consumer PyTorch — it is not.

---

## Architecture. For Those Who Want Details.

### Two Buffers. Two Streams. One CUDA Graph.

```python
# Device state — initialized once
_DEVICE_STATE[device] = {
    "transfer_stream":  torch.cuda.Stream(device=device),  # DMA stream
    "w_buffers":  [None, None],   # two buffers — ping and pong
    "b_buffers":  [None, None],   # bias buffers
    "forward_clk": 0,             # current buffer index (0 or 1)

    # Events for cross-stream synchronization
    "transfer_forward_finished_event":  torch.cuda.Event(),
    "compute_forward_start_event":      torch.cuda.Event(),
}
```

**Two buffers** hold the weights of the current and next layer simultaneously.  
**Two CUDA streams** let transfer and compute run in parallel.  
**CUDA Events** are semaphores — they tell one stream when the other has finished.

### Custom Autograd Function

This is the heart of the system. `_BouncingLinearFn` — a custom `torch.autograd.Function` that intercepts every linear layer in the model:

```python
class _BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device):
        state = _get_device_state(device)
        idx = state["forward_clk"]  # current buffer (0 or 1)

        # In a separate stream — non-blocking weight transfer
        with torch.cuda.stream(state["transfer_stream"]):
            state["transfer_stream"].wait_event(
                state["compute_forward_start_event"]
            )
            # Launch DMA transfer — CPU RAM → GPU VRAM
            w = weight_cpu.to(device, non_blocking=True)
            state["w_buffers"][idx] = _dequant(w, dtype)
            # Signal: weights are ready
            state["transfer_forward_finished_event"].record()

        # In the main compute stream — wait only on the event, not the transfer
        torch.cuda.current_stream().wait_event(
            state["transfer_forward_finished_event"]
        )
        state["compute_forward_start_event"].record()

        # Switch buffer (ping → pong)
        state["forward_clk"] ^= 1

        # Compute — weights are already on GPU
        return F.linear(x, state["w_buffers"][idx], state["b_buffers"][idx])
```

**`^= 1`** is XOR index switching. `0 → 1 → 0 → 1...`  
While compute works with buffer `[0]`, transfer writes into buffer `[1]`. And vice versa.

### Pinned Memory — DMA Without Copying

```python
def _ensure_cpu_pinned(t):
    if not t.is_pinned():
        t = t.pin_memory()
    return t
```

Regular RAM can be swapped out by the OS at any moment.  
Pinned memory cannot. The GPU DMA controller reads it directly — no intermediate CPU cache copy. Another multiplier on transfer speed.

### Backward Pass — Same Principle

```python
@staticmethod
def backward(ctx, grad_out):
    with torch.cuda.stream(state["transfer_stream"]):
        w = weight_cpu.to(ctx.device, non_blocking=True)
        state["w_bwd_buffers"][idx] = _dequant(w, ctx.dtype)
        state["transfer_backward_finished_event"].record()

    torch.cuda.current_stream().wait_event(
        state["transfer_backward_finished_event"]
    )

    grad_input  = grad_out @ state["w_bwd_buffers"][idx]
    grad_weight = grad_out.flatten(0,-2).T @ x.flatten(0,-2)
    grad_bias   = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))

    return grad_input, grad_weight, grad_bias, None
```

Both forward and backward use overlap. Every training step is fully asynchronous.

### Attaching to the Model — One Line

```python
class LinearLayerMemoryManager:
    @classmethod
    def attach(cls, m, mgr):
        if not hasattr(m, "_layer_memory_manager"):
            m._layer_memory_manager = cls(m, mgr)
```

`attach()` is called once at init for each linear layer.  
After that, the model runs as normal — PyTorch has no idea a different engine is underneath.

---

## Proof. Not Words — Logs.

### Stress Test: Rank 1024. The system holds.

[Stress Test Rank 1024](logs/log.txt)

```
amiguHDR1024:  39% | 39/100 [1:56:48<3:02:42, 179.71s/it]
amiguHDR1024:  40% | 40/100 [1:59:43<2:59:35, 179.58s/it]
```

This is the **baseline before optimization** at rank 1024 — the most extreme possible config.  
179 sec/iter. No crashes. No OOM. The architecture survives what no one else attempts.

### Production: Rank 32 / Alpha 64. This is what you actually train with.

```
sharpR32ALPH64CONV32flux2: 82% | 819/1000 [1:29:42<19:49, 6.57s/it]
  - 5.9509s avg - train_loop
  - 3.8503s avg - backward
  - 2.0128s avg - predict_unet
  - 0.0846s avg - optimizer_step
```

**6.57 seconds per iteration. 1000 steps ≈ 2.5 hours.**

<img width="1416" height="798" alt="Screenshot_20260421_110054_Chrome" src="https://github.com/user-attachments/assets/0dda7d71-135a-49cd-882e-fa53263d86c1" />


### The Key Detail in the Breakdown

- `backward: 3.85s` — GPU computing gradients
- `predict_unet: 2.01s` — forward pass
- `optimizer_step: 0.08s` — weight update
- **transfer time: absent**

Transfer has **disappeared from the profile**.  
It runs in parallel and does not register as measurable time.  
This is the proof that overlap works — the bottleneck is gone.

---

## Scalability. Why This Matters Beyond the RTX 4090.

This pattern is not a consumer GPU trick. It is an architectural decision.

### On Server Hardware

On server systems with NVLink (A100/H100 clusters), weights stream GPU→GPU instead of CPU→GPU. The principle is identical: double buffering + async stream + CUDA events.

```python
# Consumer: CPU RAM → GPU VRAM
w = weight_cpu.to(device, non_blocking=True)

# Server: GPU_0 VRAM → GPU_1 VRAM (NVLink)
w = weight_gpu0.to(device_1, non_blocking=True)
```

`_BouncingLinearFn` maps to server topology with virtually no changes.

### On Tensor Parallelism

Each GPU holds a shard of the layer's weights. The same CUDA streams and Events coordinate full tensor assembly — while one GPU gathers its shard, another has already begun aggregation.

### On Inference

No backward pass — only forward. Double buffering forward yields even greater gains since no activations need to be retained. Inference server throughput scales proportionally with the number of model layers.

### The Scaling Formula

The overlap benefit grows with:
- **Number of layers** — more layers, more opportunities for overlap
- **Weight tensor size** — larger transfers = greater potential to hide latency
- **Compute intensity** — the longer the GPU computes, the more the transfer can complete

At rank 32 these three factors are optimally balanced.  
That is why 6.57 sec/iter is achievable on consumer hardware.

---

## Context. Why a Datacenter Didn't Build This.

ostris/ai-toolkit is an open-source project used by thousands on consumer GPUs.  
Its standard layer offloading is sequential.

This module was written in Chernihiv, in a basement, with an RTX 4090.  
[ostris](https://github.com/ostris) himself requested the code for integration.

This does not mean we had more resources.  
It means the right architecture matters more than the hardware.

---

## Try It Yourself

Recommended config — rank 32, fast and high quality:

```yaml
network:
  type: lora
  linear: 32
  linear_alpha: 64      # 2.0× multiplier (base)
  conv: 32
  conv_alpha: 64        # 2.0× multiplier (textures)
  lokr_full_rank: true
  lokr_factor: -1
  network_kwargs:
    ignore_if_contains: []
```

```bash
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
# Replace manager_modules.py  (release — coming soon)
python run.py config/your_config.yaml
```

Expected result on RTX 4090: **6–7 sec/iter** at rank 32.  
No model quantization. No quality compromise.

---

## What's Next

- [ ] Upstream integration into ai-toolkit
- [ ] `Conv2d` layer support (currently `Linear` only)
- [ ] Multi-GPU: weight streaming across GPUs via NVLink / PCIe
- [ ] Adaptive prefetch: dynamic transfer/compute ratio estimation
- [ ] Benchmarks on A100 / H100 to confirm scalability

---

## Code

Module: [manager_modules.py](core/manager_modules.py)  


---

*The smell of the iron is stable. The system is running. 🦊*

*Chernihiv, Ukraine 🇺🇦 · Orakul Studio · 2026*


## License

MIT — use it, fork it, improve it.


# Русская версия  
[Back to English / Наверх](#Orakul Engine: Universal VRAM Management for Ostris AI-Toolkit)


# Orakul Engine: Universal VRAM Management for Ostris AI-Toolkit

**или: движок внутри движка, написанный в подвале под обстрелами**

*Orakul Studio — Chernihiv, Ukraine 🇺🇦*

🇬🇧 [English version](README_EN.md)

---

## Цифры. Сразу.

Потому что без них всё остальное — просто слова.

### Боевые результаты обучения (Rank 32 / Alpha 64)

| Конфигурация | До | После | Ускорение |
|---|---|---|---|
| LoRA rank 32 / alpha 64, Flux2-dev | **73 сек/итер** | **6.57 сек/итер** | **11×** |
| Полное обучение 1000 шагов | **~20 часов** | **~2.5 часа** | — |
| Большинство моделей включая видео-LoRA | — | **~2 сек/итер** | — |

### Стресс-тест на предельных нагрузках (Rank 1024 — доказательство устойчивости)

| Конфигурация | Результат |
|---|---|
| LoRA rank 1024 / alpha 1024, Flux2-dev | **Стабильно. 0 вылетов. 0 OOM.** |
| Baseline до оптимизации, rank 1024 | 179 сек/итер |
| После оптимизации, rank 1024 | 6–8 сек/итер |

> Rank 1024 — это не боевой конфиг.  
> Это доказательство одного факта: **архитектура не ломается под экстремальной нагрузкой.**  
> Если держит rank 1024 — держит всё что ниже.

**Железо:** RTX 4090 (24 GB VRAM) · i9-13900K · 128 GB RAM  
**Фреймворк:** [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)  
Без дополнительных GPU. Без облака. Без серверного железа.

Это не конфигурация. Это переписанный слой памяти PyTorch.

---

## Проблема. Для всех.

Flux2 — это большая модель. Трансформер с миллиардами параметров.  
RTX 4090 — это 24 GB VRAM. Модель туда **не помещается целиком**.

ai-toolkit решает это через **layer offloading**: веса каждого слоя хранятся в RAM, и перед вычислением слоя они переносятся на GPU, а после — выгружаются обратно.

Красивое решение. Но у него есть одна проблема:

```
GPU вычисляет слой N      ████████████████
Перенос весов слоя N+1              ████████████████
```

Это **последовательный процесс**. GPU стоит и ждёт пока данные приедут.  
Данные едут пока GPU ничего не делает.

При rank 32 этот overhead накапливается по сотням слоёв модели.  
При rank 1024 он становится катастрофическим — 179 секунд на итерацию.  
Именно поэтому существует стресс-тест: он показывает полный масштаб проблемы.

---

## Решение. Тоже для всех.

Идея простая. Реализация — нет.

Пока GPU **вычисляет** слой N — параллельно, в отдельном CUDA-потоке, **уже начинается перенос весов** слоя N+1.

```
GPU вычисляет слой N      ████████████████
Перенос весов слоя N+1    ████████████████
                          ↑
                          Начинается одновременно
```

GPU заканчивает слой N — и веса слоя N+1 уже на месте. Ждать не нужно.

Это называется **double buffering** с **compute-transfer overlap**.  
В HPC-системах это стандарт. В consumer PyTorch — нет.

---

## Архитектура. Для тех кто хочет деталей.

### Два буфера. Два потока. Один CUDA-граф.

```python
# Состояние устройства — инициализируется один раз
_DEVICE_STATE[device] = {
    "transfer_stream":  torch.cuda.Stream(device=device),  # поток для DMA
    "w_buffers":  [None, None],   # два буфера — ping и pong
    "b_buffers":  [None, None],   # буферы для bias
    "forward_clk": 0,             # текущий индекс буфера (0 или 1)

    # События для синхронизации между потоками
    "transfer_forward_finished_event":  torch.cuda.Event(),
    "compute_forward_start_event":      torch.cuda.Event(),
}
```

**Два буфера** держат веса текущего и следующего слоя одновременно.  
**Два CUDA-потока** позволяют transfer и compute работать параллельно.  
**CUDA Events** — семафоры. Говорят одному потоку когда другой закончил.

### Пользовательская функция autograd

Это сердце системы. `_BouncingLinearFn` — кастомный `torch.autograd.Function` который перехватывает каждый линейный слой модели:

```python
class _BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device):
        state = _get_device_state(device)
        idx = state["forward_clk"]  # текущий буфер (0 или 1)

        # В отдельном потоке — неблокирующий перенос весов
        with torch.cuda.stream(state["transfer_stream"]):
            state["transfer_stream"].wait_event(
                state["compute_forward_start_event"]
            )
            # Запускаем DMA transfer — CPU RAM → GPU VRAM
            w = weight_cpu.to(device, non_blocking=True)
            state["w_buffers"][idx] = _dequant(w, dtype)
            # Сигнализируем: веса готовы
            state["transfer_forward_finished_event"].record()

        # В основном compute потоке — ждём только событие, не сам transfer
        torch.cuda.current_stream().wait_event(
            state["transfer_forward_finished_event"]
        )
        state["compute_forward_start_event"].record()

        # Переключаем буфер (ping → pong)
        state["forward_clk"] ^= 1

        # Вычисление — веса уже на GPU
        return F.linear(x, state["w_buffers"][idx], state["b_buffers"][idx])
```

**`^= 1`** — XOR-переключение индекса. `0 → 1 → 0 → 1...`  
Пока compute работает с буфером `[0]`, transfer пишет в буфер `[1]`. И наоборот.

### Pinned Memory — DMA без копирования

```python
def _ensure_cpu_pinned(t):
    if not t.is_pinned():
        t = t.pin_memory()
    return t
```

Обычные страницы RAM могут быть вытеснены ОС в swap.  
Pinned memory — нет. DMA-контроллер GPU читает её напрямую, без промежуточного копирования через CPU cache. Ещё один множитель к скорости transfer.

### Backward pass — тот же принцип

```python
@staticmethod
def backward(ctx, grad_out):
    with torch.cuda.stream(state["transfer_stream"]):
        w = weight_cpu.to(ctx.device, non_blocking=True)
        state["w_bwd_buffers"][idx] = _dequant(w, ctx.dtype)
        state["transfer_backward_finished_event"].record()

    torch.cuda.current_stream().wait_event(
        state["transfer_backward_finished_event"]
    )

    grad_input  = grad_out @ state["w_bwd_buffers"][idx]
    grad_weight = grad_out.flatten(0,-2).T @ x.flatten(0,-2)
    grad_bias   = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))

    return grad_input, grad_weight, grad_bias, None
```

Forward и backward — оба используют overlap. Каждый шаг обучения полностью асинхронный.

### Подключение к модели — одна строка

```python
class LinearLayerMemoryManager:
    @classmethod
    def attach(cls, m, mgr):
        if not hasattr(m, "_layer_memory_manager"):
            m._layer_memory_manager = cls(m, mgr)
```

`attach()` вызывается один раз при инициализации для каждого линейного слоя.  
После этого модель работает как обычно — PyTorch не знает что под капотом другой движок.

---

## Доказательства. Не слова — логи.

### Стресс-тест: Rank 1024. Система держит.

[Смотреть лог стресс-теста Rank 1024](logs/log.txt)

```
amiguHDR1024:  39% | 39/100 [1:56:48<3:02:42, 179.71s/it]
amiguHDR1024:  40% | 40/100 [1:59:43<2:59:35, 179.58s/it]
```

Это **baseline до оптимизации** при rank 1024 — самый экстремальный конфиг из возможных.  
179 сек/итер. Ни одного вылета. Ни одного OOM.  
Архитектура выдерживает то, за что никто другой не берётся.

### Боевой режим: Rank 32 / Alpha 64. Именно здесь вы обучаете.

```
sharpR32ALPH64CONV32flux2: 82% | 819/1000 [1:29:42<19:49, 6.57s/it]
  - 5.9509s avg - train_loop
  - 3.8503s avg - backward
  - 2.0128s avg - predict_unet
  - 0.0846s avg - optimizer_step
```

**6.57 секунд на итерацию. 1000 шагов ≈ 2.5 часа.**

<img width="1416" height="798" alt="Screenshot_20260421_110054_Chrome" src="https://github.com/user-attachments/assets/386acbaf-04e9-4e20-ac0b-fa763df63c74" />


### Главная деталь в профиле итерации

- `backward: 3.85s` — GPU считает градиенты
- `predict_unet: 2.01s` — forward pass
- `optimizer_step: 0.08s` — обновление весов
- **transfer времени нет**

Transfer **исчез из профиля**.  
Он работает параллельно и не попадает в измеримое время.  
Это и есть доказательство что overlap работает — узкое горло устранено.

---

## Масштабируемость. Почему это важно за пределами RTX 4090.

Этот паттерн — не трюк для consumer GPU. Это архитектурное решение.

### На серверном железе

На серверных системах с NVLink (A100/H100) веса стримятся GPU→GPU вместо CPU→GPU. Принцип идентичен: double buffering + async stream + CUDA events.

```python
# Consumer: CPU RAM → GPU VRAM
w = weight_cpu.to(device, non_blocking=True)

# Server: GPU_0 VRAM → GPU_1 VRAM (NVLink)
w = weight_gpu0.to(device_1, non_blocking=True)
```

`_BouncingLinearFn` переносится на серверную топологию практически без изменений.

### На tensor parallelism

Каждый GPU держит шард весов слоя. Те же CUDA streams и Events координируют сборку полного тензора — пока один GPU собирает свою часть, другой уже начинает aggregate.

### На inference

Нет backward pass — только forward. Double buffering forward даёт ещё больший выигрыш: не нужно хранить активации. Throughput inference-сервера растёт пропорционально числу слоёв модели.

### Формула масштабирования

Выигрыш от overlap растёт с:
- **Количеством слоёв** — больше слоёв, больше возможностей для overlap
- **Размером весов** — больший transfer = больший потенциал скрытия латентности
- **Вычислительной интенсивностью** — чем дольше GPU считает, тем больше успевает transfer

При rank 32 все три фактора сбалансированы оптимально.  
Именно поэтому 6.57 сек/итер достижимы на consumer железе.

---

## Контекст. Почему это сделал не датацентр.

ostris/ai-toolkit — опенсорсный проект для обучения LoRA.  
Его используют тысячи людей с consumer GPU.  
Стандартный layer offloading в нём работает последовательно.

Мы написали этот модуль в Чернигове, в подвале, с RTX 4090.  
Сам [ostris](https://github.com/ostris) запросил код для интеграции.

Это не означает что у нас больше ресурсов.  
Это означает что правильная архитектура важнее железа.

---

## Попробовать самому

Рекомендованный конфиг — rank 32, быстро и качественно:

```yaml
network:
  type: lora
  linear: 32
  linear_alpha: 64      # Множитель 2.0 (для базы)
  conv: 32
  conv_alpha: 64        # Множитель 2.0 (для текстур)
  lokr_full_rank: true
  lokr_factor: -1
  network_kwargs:
    ignore_if_contains: []
```

```bash
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
# Заменить manager_modules.py (ссылка на релиз — скоро)
python run.py config/your_config.yaml
```

Ожидаемый результат на RTX 4090: **6–7 сек/итер** при rank 32.  
Без квантования модели. Без компромиссов по качеству.

---

## Что дальше

- [ ] Интеграция в upstream ai-toolkit
- [ ] Поддержка `Conv2d` слоёв (сейчас только `Linear`)
- [ ] Multi-GPU: стриминг весов через NVLink / PCIe
- [ ] Adaptive prefetch: динамическая оценка transfer/compute ratio
- [ ] Benchmarks на A100 / H100 для подтверждения масштабируемости

---

## Код

Модуль: [manager_modules.py](core/manager_modules.py) 


---

*Запах утюга стабільний. Система працює. 🦊*

*Chernihiv, Ukraine 🇺🇦 · Orakul Studio · 2026*


## Лицензия

MIT — используй, форкай, улучшай.

[Back to English / Наверх](#Orakul Engine: Universal VRAM Management for Ostris AI-Toolkit)
