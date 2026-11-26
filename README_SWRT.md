# Программная трассировка (SWRT) для VulkanProxy

## Как собрать
Положите рядом с dllmain.cpp: `swrt_build.h swrt_spv.h swrt_proxy.h swrt_rq.h swrt_rq_spv.h`
(`dllmain.cpp` их подключает сам). Замените `VulkanProxy.ini`. Новых зависимостей нет.
`swrt_spv.h` / `swrt_rq_spv.h` сгенерированы из `swrt_dev/shaders/*.comp`
(`python gen_spv.py`, `python gen_rq.py`, нужен glslangValidator) — трогать их руками не надо.

## Что сделано
1. vkCreateBuffer: RT-флаги usage (AS storage/build input/SBT) заменяются на STORAGE|DEVICE_ADDRESS.
2. vkCreateDevice: принудительно включаются shaderInt64, bufferDeviceAddress, scalarBlockLayout.
3. AS стали настоящими: запись на буфер игры, адрес = реальный device address буфера.
4. vkCmdBuildAccelerationStructuresKHR -> GPU-построение BVH (LBVH: morton + bitonic sort + Karras + refit)
   прямо в командном буфере игры; её compute-пайплайн и push constants восстанавливаются.
5. Размеры AS/scratch под наш формат, vkCmdCopyAccelerationStructureKHR, query-пулы compacted size (эмуляция на CPU).
6. Шейдеры: OpRayQuery* заменяются вызовами библиотеки обхода BVH (SPIR-V-линковка в память шейдера),
   AS-дескрипторы превращаются в storage buffer (layout, pool, update).
7. Опция `SoftwareRT=1`; при неудаче переписывания шейдер падает на старое вырезание лучей.
8. HeavyComputeWords сравнивается с исходным размером шейдера.

## Проверено (Linux, lavapipe, сверка с перебором на CPU)
BVH-построение + обход: 20000 лучей, 0 расхождений (t, инстанс, примитив, барицентрика),
индексы u16/u32/без индексов, трансформы инстансов, маски, cull-флаги.
Переписанные ray-query шейдеры (в т.ч. Private-переменная, вызов из функции, Confirm/Terminate,
матрицы, buffer_reference в самом шейдере) проходят spirv-val и дают те же результаты.
dllmain.cpp проходит проверку синтаксиса mingw (с заглушками для generated-заголовка).

## НЕ проверено (нужен ваш запуск)
Реальная GPU/драйвер R7 260X, компиляция ядер и переписанных шейдеров драйвером, скорость,
причина DEVICE_LOST из лога, сам Doom. Смотрите в логе строки `SWRT:`.

## Известные ограничения
- Все кандидаты считаются непрозрачными (alpha-test геометрия даёт тени/попадания как сплошная).
- AABB-геометрия, вершины не f32/f16/snorm16, индексы uint8 — игнорируются (в логе `SWRT: unsupported`).
- Descriptor update templates и push descriptors для AS не конвертируются.
- vkCmdTraceRays / RT-пайплайны не эмулируются (только ray query).
- После vkCmdCopyAccelerationStructureKHR внутренний self-адрес TLAS устарел (BLAS копировать можно).
- Шейдеры на 100k+ слов компилятор GCN2 может не осилить независимо от прокси.
