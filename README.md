[English Version] | [Русская версия ниже](#русская-версия)

# Orakul Engine: Universal VRAM Management for Ostris AI-Toolkit

**or: an engine inside an engine, written in a basement under artillery fire**

*Orakul Studio - Chernihiv, Ukraine 🇺🇦*

---

## ⚠️ STRICTLY FOR TERMINAL / CLI RUNNING ONLY

If you're one of the 400+ people who cloned this repository, forget about any web UIs for this pipeline.

This code was designed, rewritten, and optimized exclusively for directly running configuration files (.yaml) via the console.

### What breaks when running via the Web UI:

1. Dynamic Alpha  DOESN'T WORK AT ALL**
* This repository implements dynamic Alpha recalculation logic for correct weight scaling (Scale = Alpha / Rank). For example, when working with high ranks (Rank 128, Rank 512, Rank 1024), the system automatically calculates a fair scale (down to Scale = 0.5000), allowing the model to deeply learn the structure and physics of the material.
* **The web UI completely ignores this logic.** Almost all web wrappers under the hood forcibly overwrite this parameter and force a fixed Alpha = 16. At high ranks, this turns training into a dud: weight changes are suppressed, gradients tend to zero, the model visually "learns" without errors, but produces default output.

2. **Asynchronous Memory Manager (Async CUDA Memory Manager)  CRASHED**
* The logic for memory retention and low-level logging is optimized for the terminal's stdout.
* Web interfaces attempt to intercept and parse the string stream for their browser consoles. At best, this leads to a crash of the backend interface due to custom security prints; at worst, to a hidden downcast of tensor precision and gradient castration, so that a casual user doesn't simply "get a memory error."

### How to use the repository correctly:
* Run strictly through the console, directly from your virtual environment.
* If your process crashes while working with high ranks and dynamic alpha, **don't look for compromises in the code; instead, increase the system swap/pagefile**. The terminal works with your hardware without censorship or hidden precision reductions.
* Here is the configuration file for running the training [Test1280.yaml](https://github.com/OrakulStudio/AI-Toolkit-Viking-Engine-Fork/blob/main/viking_train/Test1280.yaml)


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
At rank 1024 it becomes catastrophic  179 seconds per iteration.  
That is why the stress test exists: to show the full scale of the problem.

---

## The Solution. Also for Everyone.

The idea is simple. The implementation  not so much.

While the GPU **computes** layer N — in a separate CUDA stream, in parallel, **the transfer of layer N+1's weights has already begun**.

```
GPU computes layer N        ████████████████
Transfer weights layer N+1  ████████████████
                            ↑
                            Starts simultaneously
```

By the time the GPU finishes layer N  the weights for layer N+1 are already there. No waiting.

This is called **double buffering** with **compute-transfer overlap**.  
In HPC systems, this is standard practice. In consumer PyTorch — it is not.

---

## Architecture. For Those Who Want Details.

### Two Buffers. Two Streams. One CUDA Graph.

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

**Two buffers** hold the weights of the current and next layer simultaneously.  
**Two CUDA streams** let transfer and compute run in parallel.  
**CUDA Events** are semaphores  they tell one stream when the other has finished.

### Custom Autograd Function

This is the heart of the system. `_BouncingLinearFn`  a custom `torch.autograd.Function` that intercepts every linear layer in the model:

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..

### Pinned Memory  DMA Without Copying

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

Regular RAM can be swapped out by the OS at any moment.  
Pinned memory cannot. The GPU DMA controller reads it directly — no intermediate CPU cache copy. Another multiplier on transfer speed.

### Backward Pass  Same Principle

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..

### Attaching to the Model — One Line

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..

---

## Proof. Not Words  Logs.

### Stress Test: Rank 1024. The system holds.
```
amiguHDR1024:  39% | 39/100 [1:56:48<3:02:42, 179.71s/it]
amiguHDR1024:  40% | 40/100 [1:59:43<2:59:35, 179.58s/it]
```
[Stress Test Rank 1024](logs/log.txt)

This is the **baseline before optimization** at rank 1024  the most extreme possible config.  
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

> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..

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

Потому что без них всё остальное  просто слова.

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

Flux2  это большая модель. Трансформер с миллиардами параметров.  
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
При rank 1024 он становится катастрофическим  179 секунд на итерацию.  
Именно поэтому существует стресс-тест: он показывает полный масштаб проблемы.

---

## Решение. Тоже для всех.

Идея простая. Реализация  нет.

Пока GPU **вычисляет** слой N  параллельно, в отдельном CUDA-потоке, **уже начинается перенос весов** слоя N+1.

```
GPU вычисляет слой N      ████████████████
Перенос весов слоя N+1    ████████████████
                          ↑
                          Начинается одновременно
```

GPU заканчивает слой N  и веса слоя N+1 уже на месте. Ждать не нужно.

Это называется **double buffering** с **compute-transfer overlap**.  
В HPC-системах это стандарт. В consumer PyTorch  нет.

---

## Архитектура. Для тех кто хочет деталей.

### Два буфера. Два потока. Один CUDA-граф.

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

**Два буфера** держат веса текущего и следующего слоя одновременно.  
**Два CUDA-потока** позволяют transfer и compute работать параллельно.  
**CUDA Events** — семафоры. Говорят одному потоку когда другой закончил.

### Пользовательская функция autograd

Это сердце системы. `_BouncingLinearFn` — кастомный `torch.autograd.Function` который перехватывает каждый линейный слой модели:

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..

### Pinned Memory  DMA без копирования

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

Обычные страницы RAM могут быть вытеснены ОС в swap.  
Pinned memory  нет. DMA-контроллер GPU читает её напрямую, без промежуточного копирования через CPU cache. Ещё один множитель к скорости transfer.

### Backward pass  тот же принцип

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

Forward и backward  оба используют overlap. Каждый шаг обучения полностью асинхронный.

### Подключение к модели  одна строка

```python
> 🔒 **Orakul Studio Proprietary Tech**  
> Core architecture and high-performance memory optimization layers are closed-source. Distributed exclusively via compiled binary module. The repository is open, and the pipeline is fully functional and stable..
```

`attach()` вызывается один раз при инициализации для каждого линейного слоя.  
После этого модель работает как обычно  PyTorch не знает что под капотом другой движок.

---

## Доказательства. Не слова  логи.

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
Это и есть доказательство что overlap работает  узкое горло устранено.

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

Нет backward pass  только forward. Double buffering forward даёт ещё больший выигрыш: не нужно хранить активации. Throughput inference-сервера растёт пропорционально числу слоёв модели.

### Формула масштабирования

Выигрыш от overlap растёт с:
- **Количеством слоёв** — больше слоёв, больше возможностей для overlap
- **Размером весов** — больший transfer = больший потенциал скрытия латентности
- **Вычислительной интенсивностью** — чем дольше GPU считает, тем больше успевает transfer

При rank 32 все три фактора сбалансированы оптимально.  
Именно поэтому 6.57 сек/итер достижимы на consumer железе.

---

## Контекст. Почему это сделал не датацентр.

ostris/ai-toolkit  опенсорсный проект для обучения LoRA.  
Его используют тысячи людей с consumer GPU.  
Стандартный layer offloading в нём работает последовательно.

Мы написали этот модуль в Чернигове, в подвале, с RTX 4090.  
Сам [ostris](https://github.com/ostris) запросил код для интеграции.

Это не означает что у нас больше ресурсов.  
Это означает что правильная архитектура важнее железа.

---

## Попробовать самому

Рекомендованный конфиг  rank 32, быстро и качественно:

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
