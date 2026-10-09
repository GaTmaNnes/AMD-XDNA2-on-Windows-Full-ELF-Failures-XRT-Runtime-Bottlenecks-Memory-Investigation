# AMD XDNA2 / Windows — Technical Failure Analysis and Cross-Issue Investigation

**Date:** October 9, 2026  
**Platform:** AMD Strix Point, XDNA2 NPU, Windows 11, XRT-MCDM  
**Reported driver:** 3761  
**Primary issue:** https://github.com/Xilinx/mlir-aie/issues/3910  
**Related performance investigation:** https://github.com/ROCm/FastFlowLM/issues/636

**Status:** Evidence-based investigation and proposed validation plan. No universal root cause has been established.

---

## 1. Executive Summary

This report investigates the relationships among several publicly documented problems affecting AMD XDNA2 development, including:

- Native Full-ELF loading failures on Windows.
- Differences between Windows and Linux NPU execution.
- Host-buffer limits and DDR address translation.
- Memory-group mismatches causing silent data corruption.
- Multi-column control-packet configuration timeouts.
- Dynamic runtime-sequence limitations.
- XRT launcher initialization and data-transfer overhead.
- Performance implications for low-latency LLM decoding.

### Principal finding

These issues are not independent at the architectural level: they involve overlapping components of the host runtime, context management, binary metadata, buffer handling, and device command submission.

However, shared components do not imply shared defects.

The strongest conclusion supported by current evidence is that **host-runtime behavior and execution-path compatibility must be validated separately from AIE kernel computation**.

In particular, a successful legacy XCLBIN execution does not establish that the native Full-ELF loader is compatible with the same artifacts.

Likewise, a completed NPU command does not necessarily establish that correct memory was accessed.

### Key conclusions

1. Windows-specific XDNA2 execution failures have already been formally acknowledged in MLIR-AIE's CI tracking.
2. The Full-ELF loading failure reported in #3910 remains distinguishable from those existing defects.
3. Multiple different ELF-related execution paths exist, and their success or failure must not be conflated.
4. Incorrect buffer address translation or memory-group selection can produce aborts, stale results, or silent corruption.
5. A separate AMD Triton-XDNA investigation demonstrates that repeatedly rebuilding runtime state can dominate launch latency.
6. Persistent contexts, validated instruction streams, pooled buffers, and dynamic runtime sequencing are promising optimization directions, but their performance gains must be measured rather than assumed.

---

## 2. Evidence Classification

The following classification is used throughout:

| Classification | Meaning |
|---|---|
| VERIFIED UPSTREAM | Explicitly documented in a public upstream issue, source, or maintainer discussion |
| REPORTED / REPRODUCED | Observed and reported by the local investigator, without independent reproduction by AMD |
| CORRELATED | Components or behaviors are technically related, but no shared root cause is demonstrated |
| HYPOTHESIS | Proposed explanation requiring targeted testing |
| PROPOSED | Engineering action or test not yet demonstrated as successful |

This distinction is essential when combining investigations that use different operating systems, XRT versions, firmware revisions, and hardware configurations.

---

## 3. Primary Failure: Native Full-ELF Loading on Windows

**Source:** https://github.com/Xilinx/mlir-aie/issues/3910

**Classification:** REPORTED / REPRODUCED

### Environment

- Windows 11
- AMD Strix Point XDNA2
- NPU driver 3761
- Native MSVC host applications
- IRON / MLIR-AIE-generated ELF artifacts
- Exact XRT and compiler revisions not yet supplied in the published issue

### Observations

Three generated ELF artifacts were tested:

| Artifact size | Native Full-ELF loading |
|---|---|
| 157 KB | Failure |
| 73 KB | Failure |
| 11.8 KB | Failure |

All three produced the reported error:

`Invalid instruction buffer size`

The investigated API path involves:

`xrt::elf → xrt::hw_context(elf) → xrt::module(elf) → xrt::module(parent, hwctx)`

The exact API call that first produces the error has not yet been identified in the published reproducer.

### Positive control

The `.ctrltext.0` instructions extracted from IRON-generated ELF artifacts can execute through an existing XCLBIN-backed context and the legacy `MLIR_AIE` kernel path.

Additional successful local experiments include Q/K/V GEMVs, INT4 GEMVs, and fused post-attention operations.

### What this establishes

- The target device can execute custom AIE instructions in at least one supported runtime configuration.
- The reported failure is associated with the native Full-ELF path or its inputs.
- The failure is not sufficient evidence of defective AIE compute hardware.

### What this does not establish

- That the Full-ELF artifacts are correctly packaged.
- That the native Windows loader is defective.
- That missing `.ctrlpkt.pm` metadata causes the failure.
- That the same ELF loads successfully under Linux.
- That the loader failure and previously reported CU-mask rejection share a root cause.

### Main hypotheses

**H1 — Artifact or metadata mismatch**

The Full-ELF loader may expect sections, descriptors, alignment, instruction-size fields, or other metadata not provided in the tested artifacts.

**H2 — Windows-specific loader or driver incompatibility**

The Windows XRT-MCDM implementation may process the ELF differently from the Linux implementation.

**H3 — API usage or version mismatch**

The generated ELF, XRT SDK, runtime libraries, and driver may not implement matching versions of the expected ABI or module-loading contract.

**H4 — Context / CU resolution incompatibility**

The ELF may be accepted at one stage but become incompatible when an executable context or compute-unit identity is resolved.

These hypotheses should be investigated independently.

---

## 4. Windows-Only NPU Failures Already Tracked Upstream

**Sources:**

- https://github.com/Xilinx/mlir-aie/issues/3439
- https://github.com/Xilinx/mlir-aie/issues/3438
- https://github.com/Xilinx/mlir-aie/issues/3437
- https://github.com/Xilinx/mlir-aie/pull/3441

**Classification:** VERIFIED UPSTREAM

### 4.1 Windows CI tracking (#3439)

MLIR-AIE maintainers introduced native-Windows NPU CI coverage and identified tests that passed under Linux but failed under Windows.

Two distinct categories were tracked:

1. Invalid code generation for dynamic runtime-sequence arguments.
2. NPU command aborts involving larger numbers of host buffers.

The code-generation case was fixed by PR #3441.

The host-buffer problem remained separately tracked in #3438 in the reviewed records.

### 4.2 Host-buffer aborts (#3438)

Affected tests include:

- `test/python/npu-xrt/test_jit_many_args.py`
- `test/npu-xrt/many_buffers`

Observed failures:

- `ERT_CMD_STATE_ABORT`
- Command status 6

The original investigation highlighted a firmware and compiler address-patching contract involving the first five DDR buffers.

An additional maintainer investigation identified a practical Windows limit of seven user-declared host buffers plus the instruction buffer in the tested implementation.

These numbers describe different aspects of the implementation and should not be treated as interchangeable.

### 4.3 Address-translation mechanism

The upstream report identifies:

- `kNumFirmwareTranslatedArgs = 5`
- `kDDRAIEAddrOffset = 0x80000000`
- `aie-npu-fold-ddr-addr-offset`
- `AIETargetNPU.cpp`
- `appendAddressPatch`

The hypothesis documented by the maintainers is that the Windows firmware does not reproduce the same initial DDR-address translation behavior as the Linux firmware.

### Connection to #3910

**Classification:** CORRELATED

Both investigations involve the boundary among compiled instructions, runtime metadata, and driver behavior.

However:

- #3910 reports a Full-ELF loading error.
- #3438 reports a command abort during execution.

The address-translation defect should only be considered a potential common cause if instrumentation shows that #3910 reaches the stage where BO addresses or instruction descriptors are prepared.

### Additional test

Run otherwise identical kernels with:

- One host data BO.
- Four host data BOs.
- Five host data BOs.
- Seven host data BOs.
- Eight host data BOs.

Count the instruction BO separately and record every bound kernel argument.

Use a single arena BO with validated subviews as a separate control.

**Expected result:** Establish whether the failure depends on the total number of bound BOs or on the address-patching convention.

---

## 5. Host Memory-Group Mismatch and Silent Corruption

**Source:** https://github.com/Xilinx/mlir-aie/issues/3653

**Classification:** VERIFIED UPSTREAM

### Finding

The reported Python `XRTHostRuntime` path allocated host BOs using the default `group_id=0`, while some NPU kernels declared different per-argument memory groups.

The issue provides examples with declared groups including:

- `0x10000`
- `0x20000`

The source describes incorrect results from GEMM and AXPY workloads and a larger configuration that triggered an IOMMU page fault and device failure.

A completed command could therefore still return invalid data.

### Local workaround reported upstream

The investigator restored correct results in tested cases by matching transport BO allocations to the kernel-declared memory groups.

The report also cautions that the memory-group behavior was not uniform across all tested kernels.

### Connection to the Windows investigation

**Classification:** CORRELATED

This is evidence that execution correctness depends on metadata-level compatibility between kernels and host buffers.

It is directly relevant to investigating:

- Incorrect or stale tensor contents.
- DMA faults.
- Unexpectedly successful command completion.
- BO allocations and argument binding.

It is not direct evidence that group IDs cause the Full-ELF loading error.

### Recommended validation

Record, for each buffer:

- Argument index.
- Declared `kernel.group_id()`.
- Actual allocation group.
- BO size.
- Virtual and device-visible addresses where safely observable.
- Transfer direction.
- Synchronization operations.
- Allocation lifetime.

Perform correctness checks on every output.

A `COMPLETED` status must never be used as the sole correctness oracle.

---

## 6. Multi-Column Control-Packet Configuration Failure

**Sources:**

- https://github.com/Xilinx/mlir-aie/issues/3488
- https://github.com/Xilinx/mlir-aie/pull/3461
- https://github.com/Xilinx/mlir-aie/discussions/3222

**Classification:** VERIFIED UPSTREAM / REPORTED REPRODUCTION

### Reported behavior

Under the control-packet configuration path:

`--load-pdi-to-ctrl-pkt`

Some single-column designs complete successfully, while configurations involving multiple columns hang.

The relevant runtime error is:

`ERT_CMD_STATE_TIMEOUT`

The investigation isolated column span and control-packet routing as important variables.

PR #3461 corrected part of the control-overlay generation behavior, but the upstream discussion explicitly warns against treating the issue as fully resolved without further testing.

### Relationship to Full-ELF

The multi-column problem and #3910 both involve configuration-related artifacts, but their failure stages differ.

- Full-ELF loading may fail before execution.
- Multi-column control-packet configuration times out on the device.

The underlying mechanisms may share compiler or metadata components but should not be merged into one root-cause claim.

### Recommended tests

Use the same minimal kernel topology with:

1. One compute column.
2. Two adjacent columns.
3. Two non-adjacent columns.
4. Standard PDI loading.
5. Control-packet configuration.

Compare generated packet sequences, overlay-channel selection, synchronization and completion tokens.

**Success criterion:** Bit-exact outputs, correct completion, and no timeout for every supported topology.

---

## 7. Full-ELF, Configuration ELF, and Dynamic Runtime Sequences

**Source:** https://github.com/Xilinx/mlir-aie/discussions/3222

**Classification:** VERIFIED UPSTREAM / REPORTED

### Important architectural distinction

At least four paths need separate treatment:

| Path | Characteristics |
|---|---|
| Legacy XCLBIN + instruction BO | Existing kernel/context, externally supplied instruction stream |
| Configuration ELF attached to existing context | Reconfiguration information associated with a valid XCLBIN context |
| Native Full-ELF | ELF-based context/module setup and execution |
| Dynamic TXN generation | Runtime creation or modification of executable transaction sequences |

These paths are not equivalent even if some instruction bytes are identical.

### Existing experimental evidence

The community discussion documents successful reconfiguration inside an existing context, including local 30/30 successful runs.

The same discussion documents a different ELF-native submission failure involving CU-mask/state-6 behavior on Windows.

These observations are compatible if they correspond to different context-creation and submission paths.

### Independent implementation

The `xdna-engine` developer reports using Full-ELF reconfiguration with multiple designs and per-token runtime values.

The developer also reports approximately 80–110 microseconds for specific configuration switches in that environment.

This demonstrates that certain Full-ELF operations are usable, but does not establish native Windows Full-ELF compatibility with driver 3761.

Repository:

https://github.com/atassis/xdna-engine

### Important upstream limitation

A maintainer noted that XRT's Full-ELF flow does not expose the runtime-sequence handle required by the dynamic runtime-sequence generator.

Therefore, Full-ELF and dynamically generated TXN execution cannot be assumed to compose automatically.

### Alternative architecture

A promising experimental path is:

`persistent XCLBIN`

→ `persistent hw_context / CU`

→ `runtime TXN generation or patching`

→ `instruction BO update`

→ `existing kernel submission`

This avoids depending on a newly constructed Full-ELF execution context for every dynamic change.

Nevertheless, it must be demonstrated that modified transaction sequences are valid for the resident topology and current memory mapping.

### Main validation requirements

- Correct CU identity.
- Stable context lifetime.
- Correct address relocation.
- Valid instruction size and alignment.
- Correct BO ownership.
- Synchronization before reuse.
- Bit-exact results.
- No accumulation of device state or memory leaks.

---

## 8. XRT Launch Overhead: Independent Evidence

**Sources:**

- https://github.com/amd/triton-xdna/issues/83
- https://github.com/amd/triton-xdna/issues/82

**Classification:** VERIFIED UPSTREAM

### Triton-XDNA #83

An AMD developer investigated a large difference between Triton-XDNA's HSA and XRT launchers on Strix Halo.

The published measurements included:

| Workload | XRT median | HSA median |
|---|---:|---:|
| BF16 matmul 256³ | 95.0 ms | 0.56 ms |
| BF16 matmul 512³ | 95.8 ms | 0.66 ms |

The issue attributes the gap primarily to launcher design.

The tested XRT implementation recreated device state, artifact loading, execution contexts, kernel bindings, and buffers on each invocation.

The HSA implementation reused resources across invocations.

**This is not evidence that the XRT API intrinsically adds 95 ms to every launch.**

It establishes that a stateless XRT launcher can introduce severe fixed overhead.

### Triton-XDNA #82

The separate zero-copy investigation describes copy-based data paths in both HSA and XRT launchers.

Potential improvements include:

- Reusing host-visible memory.
- Avoiding redundant host-side copies.
- Pinning suitable buffers.
- Using user-pointer BOs where supported.
- Keeping constant weights resident.

However, these optimizations introduce alignment, lifetime, and synchronization requirements.

### Connection to LLM decoding

LLM decode repeats numerous operations per token.

The latency contribution of a fixed setup cost is approximately:

`total setup overhead = operation count × per-operation setup overhead`

Reducing redundant context creation, BO allocation, and copies can therefore be valuable.

The realized gain depends on which costs exist in the actual decode implementation.

---

## 9. Local Performance Evidence and Engineering Implications

**Primary context:**

https://github.com/ROCm/FastFlowLM/issues/636

The following figures derive from the local experimental record and should not be presented as AMD-validated benchmarks.

### Recorded observations

| Metric | Local observation |
|---|---:|
| CPU Q4_0 GEMV effective bandwidth | Approximately 30–42 GB/s |
| Early NPU INT4 GEMV end-to-end estimate | Approximately 0.9–1.8 GB/s |
| Healthy CPU-oriented decode baseline | Approximately 4.3 tokens/s |
| Some NPU-heavy configurations | Approximately 0.9–1.1 tokens/s |
| XRT transport measurement | Approximately 11–12 GB/s |
| Runlist transport measurement | Approximately 0.6–2.1 GB/s |
| Fenced INT4 GEMV median | Approximately 1.1 ms |
| Fenced post-attention operation median | Approximately 21.6 ms |

These measurements were obtained from different profiling configurations.

They must not be added or compared as if all were measurements of the same execution path.

### Critical interpretation

The discrepancy between the end-to-end NPU GEMV rate and isolated fenced GEMV timings indicates that kernel time and orchestration time must be measured separately.

A 1.1 ms isolated operation does not establish that the same operation costs 1.1 ms within a complete decoder.

Similarly, the 21.6 ms post-attention measurement must be attributed to the operation that was actually instrumented. It is not proof that GEMV itself costs 21.6 ms.

### Most important optimization opportunities

1. Reduce repeated runtime setup.
2. Reuse compatible contexts and buffers.
3. Reduce unnecessary host/device transfers.
4. Isolate slow post-attention operations.
5. Fuse operations where justified by measured costs.
6. Avoid unsupported dynamic reconfiguration paths.
7. Validate device timing independently from host timing.
8. Prefer performance measurements with explicit synchronization semantics.

---

## 10. Cross-Issue Root-Cause Map

The strongest technical associations are:

| Relationship | Evidence | Interpretation |
|---|---|---|
| #3439 → #3438 | Explicit parent/sub-issue | Confirmed tracking relationship |
| #3438 ↔ Windows DDR translation | Maintainer comments and code references | Strong candidate cause for buffer aborts |
| #3653 ↔ incorrect BO groups | Reproducer and correctness workaround | Confirmed local mechanism |
| #3488 ↔ control-overlay generation | Reproducer and PR #3461 | Strong linkage |
| #3222 ↔ dynamic runtime limitations | Maintainer explanation | Confirmed interface limitation |
| Triton #83 ↔ repeated context construction | Profiling and implementation analysis | Confirmed in that launcher |
| Triton #82 ↔ redundant copies | Runtime implementation investigation | Confirmed architectural opportunity |
| #3910 ↔ native Full-ELF loading | Three reported failures | Confirmed as investigator-reported symptom |
| #3910 ↔ #3438 | Shared runtime components | Unproven common cause |
| #3910 ↔ #3488 | Related configuration machinery | Unproven common cause |
| #3910 ↔ #3653 | Memory metadata interface | Unproven common cause |

### Overall interpretation

The failures form a layered system rather than one proven defect family.

A useful decomposition is:

**Compiler / artifact generation**

→ **ELF packaging and metadata**

→ **XRT context/module construction**

→ **CU resolution**

→ **Instruction and BO binding**

→ **Driver and firmware address handling**

→ **Device configuration and execution**

→ **Synchronization and output validation**

Each layer must be tested independently before attributing a failure to another.

---

## 11. Prioritized Investigation Plan

### Phase P0 — Establish a trustworthy baseline

**Priority:** Critical

1. Freeze the current driver and toolchain configuration.
2. Record the XRT SDK version, runtime DLL versions, MLIR-AIE commit, LLVM-AIE revision, and compiler options.
3. Capture the ELF file hashes.
4. Preserve the working legacy-XCLBIN test.
5. Confirm that the known-good test is deterministic across repeated runs.

**Deliverables:**

- `environment.txt`
- `versions.txt`
- `sha256.txt`
- Baseline logs
- Exact build commands

**Exit criterion:** Another developer can identify precisely which binary and runtime versions were used.

### Phase P1 — Reproduce the native Full-ELF failure minimally

**Priority:** Critical

1. Reduce the host executable to device open, ELF loading, context creation, module construction, and minimum required launch setup.
2. Place an independent log marker before and after every API call.
3. Capture exception messages and error codes.
4. Use the smallest failing ELF.
5. Run repeated tests in a fresh process.

**Exit criterion:** Identify the first failing API call and reproduce the same error without any application-level LLM logic.

### Phase P2 — Compare with an official working ELF

**Priority:** Critical

1. Obtain a maintainer-validated native Windows Full-ELF reference.
2. Confirm compatibility with the tested Windows driver and XRT SDK.
3. Execute it unchanged.
4. Compare its API sequence and binary structure with the failing artifact.
5. Repeat under Linux if equivalent hardware and compatible software are available.

**Decision matrix:**

| Official ELF | Local ELF | Interpretation |
|---|---|---|
| PASS | FAIL | Investigate artifact generation, metadata, and API differences |
| FAIL | FAIL | Investigate loader, driver, environment, and ABI compatibility |
| PASS | PASS | Original failure may depend on versions or additional conditions |
| FAIL | PASS | Reference setup or API mismatch; investigate individually |

**Exit criterion:** A discriminating positive or negative control establishes whether the failure is specific to locally generated artifacts.

### Phase P3 — Binary and metadata analysis

**Priority:** High

Inspect and compare:

- ELF headers and program headers.
- Section names and sizes.
- Section alignments.
- `.ctrltext.0`.
- Control-packet sections, where present.
- Relocations.
- Module and kernel metadata.
- Instruction size/count representations.
- Required configuration descriptors.

Do not assume that every reference ELF is required to contain identical sections.

**Exit criterion:** A specific structural difference is correlated with a reproducible PASS/FAIL outcome.

### Phase P4 — Context and CU identity

**Priority:** High

Compare:

1. Existing XCLBIN context.
2. Config ELF attached to that context.
3. Native Full-ELF context.
4. Native ELF module resolution.
5. Submission through each supported API path.

Instrument:

- Context identity.
- Kernel/CU identity.
- Module construction outcome.
- Bound arguments.
- Instruction BO size.
- Command submission status.

**Exit criterion:** Determine whether the failure occurs before context creation, during module binding, or during submission.

### Phase P5 — Memory and address handling

**Priority:** High for runtime correctness; secondary for a pre-execution Full-ELF failure

Execute independent tests for:

- Buffer count.
- Memory group IDs.
- DDR address patching.
- BO subviews.
- BO residency.
- Synchronization.
- Instruction BO ownership.

Avoid changing all these variables simultaneously.

**Exit criterion:** Confirm valid outputs and stable command completion at known memory configurations.

### Phase P6 — Multi-column configuration

**Priority:** Medium

Test:

- Single column.
- Adjacent columns.
- Separated columns.
- Standard configuration loading.
- Control-packet loading.
- Persistent MemTile state.

**Exit criterion:** Establish the exact topology and configuration mode that triggers a timeout.

### Phase P7 — Persistent runtime and dynamic TXN

**Priority:** High for performance, after correctness

Implement and compare:

- Repeated initialization baseline.
- Persistent XCLBIN and context.
- Persistent kernel and BOs.
- Runtime instruction-BO updates.
- Dynamic TXN patching.
- Explicit synchronization.

**Exit criterion:** Correctness remains stable and repeated decode operations avoid unnecessary state rebuilding.

### Phase P8 — End-to-end LLM performance validation

**Priority:** High after P7

Profile:

- Prompt processing.
- Single-token decode.
- GEMV kernels.
- Post-attention operations.
- Host submission time.
- Actual device execution time.
- DDR data movement.
- Synchronization.
- Reconfiguration frequency.
- Peak host/device memory use.

Measure the same model, quantization, context, token count, and sampling parameters across configurations.

**Exit criterion:** Demonstrate a statistically repeatable throughput improvement without correctness regressions.

---

## 12. Requested Actions from AMD / MLIR-AIE Maintainers

The following requests are designed to be actionable rather than speculative.

### Request 1 — Reference implementation

Provide or identify a native Windows Full-ELF example tested against XRT-MCDM and driver 3761, including its exact toolchain versions.

### Request 2 — Loader diagnostics

Identify the source location producing:

`Invalid instruction buffer size`

Clarify which buffer, descriptor, or ELF field triggers that validation error.

### Request 3 — Compatibility contract

Document any Windows-specific restrictions involving:

- ELF sections.
- Alignment.
- Instruction size.
- Module construction.
- Context creation.
- Kernel/CU resolution.
- Preemption or control-packet metadata.

### Request 4 — CI coverage

Add a native-Windows test that:

1. Builds a minimal IRON ELF.
2. Loads it through the supported native Full-ELF API.
3. Creates and binds the execution context/module.
4. Executes the minimal supported kernel.
5. Verifies output correctness.
6. Reports the failing stage if the test regresses.

### Request 5 — Windows memory behavior

Clarify the supported number of bound host BOs and the intended DDR-address translation contract for the Windows driver.

### Request 6 — Dynamic runtime guidance

Clarify the preferred production-oriented architecture for repeated execution with dynamic BD addresses and a resident AIE program.

### Request 7 — Full-ELF and dynamic TXN interoperability

Document whether a supported Full-ELF interface can expose the runtime-sequence state needed for dynamic updates, or whether the recommended path remains an XCLBIN-backed instruction BO.

---

## 13. Reproduction Package Specification

A compact upstream package should contain:

```text
xdna2-windows-full-elf-repro/
├── README.md
├── environment/
│   ├── versions.txt
│   └── driver.txt
├── source/
│   ├── host.cpp
│   ├── build_windows.bat
│   └── kernel.mlir
├── artifacts/
│   ├── failing.elf
│   ├── sections.txt
│   └── sha256.txt
├── reference/
│   └── legacy-comparison.txt
├── logs/
│   ├── full-elf-failure.txt
│   └── legacy-success.txt
└── results/
    └── matrix.md
```

These names define a proposed submission structure; they do not imply that all files have already been collected.

The smallest reproducible failure should be submitted first. Large models and application-specific runtime components should remain outside the minimal reproduction.

---

## 14. Upstream Evidence and Reference URLs

### Primary reports

1. **Native Windows Full-ELF failure — #3910**  
   https://github.com/Xilinx/mlir-aie/issues/3910

2. **Windows CI failure tracker — #3439**  
   https://github.com/Xilinx/mlir-aie/issues/3439

3. **Windows host-buffer abort — #3438**  
   https://github.com/Xilinx/mlir-aie/issues/3438

4. **Windows dynamic runtime codegen — #3437**  
   https://github.com/Xilinx/mlir-aie/issues/3437

5. **Windows CI fixes — PR #3441**  
   https://github.com/Xilinx/mlir-aie/pull/3441

6. **Multi-column control-packet hang — #3488**  
   https://github.com/Xilinx/mlir-aie/issues/3488

7. **Control-overlay correction — PR #3461**  
   https://github.com/Xilinx/mlir-aie/pull/3461

8. **Host memory-group mismatch — #3653**  
   https://github.com/Xilinx/mlir-aie/issues/3653

### Runtime architecture

9. **Dynamic TXN / runtime-sequence discussion — #3222**  
   https://github.com/Xilinx/mlir-aie/discussions/3222

10. **Triton-XDNA XRT launcher latency — #83**  
    https://github.com/amd/triton-xdna/issues/83

11. **Triton-XDNA zero-copy investigation — #82**  
    https://github.com/amd/triton-xdna/issues/82

12. **xdna-engine**  
    https://github.com/atassis/xdna-engine

13. **FastFlowLM performance investigation — #636**  
    https://github.com/ROCm/FastFlowLM/issues/636

### Official documentation and implementation

14. **MLIR-AIE repository**  
    https://github.com/Xilinx/mlir-aie

15. **MLIR-AIE documentation**  
    https://xilinx.github.io/mlir-aie/

16. **Windows native-host guide**  
    https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md

17. **MLIR-AIE programming guide**  
    https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/README.md

18. **AMD XRT**  
    https://github.com/Xilinx/XRT

19. **AMD XDNA Linux driver**  
    https://github.com/amd/xdna-driver

20. **LLVM-AIE**  
    https://github.com/Xilinx/llvm-aie

21. **AMD IRON**  
    https://github.com/amd/IRON

22. **AIE-ML v2 Architecture Manual (AM027)**  
    https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2

23. **Ryzen AI documentation**  
    https://ryzenai.docs.amd.com/

---

## 15. Final Assessment

The upstream evidence demonstrates that AMD XDNA2 development currently involves several meaningful host-runtime and device-integration challenges.

Some are specific to Windows. Others concern common abstractions shared by multiple operating systems.

The most important documented categories are:

- Windows-specific host-buffer translation and command aborts.
- Incorrect host-buffer memory-group handling.
- Multi-column control-packet configuration failures.
- Limitations combining Full-ELF with dynamically generated runtime sequences.
- Excessive launcher overhead when context and buffer state are recreated unnecessarily.

The reported native Full-ELF failure in #3910 should remain a distinct issue until the failing API stage and a known-good reference are established.

**Recommended immediate action:** isolate `Invalid instruction buffer size` with a minimal native MSVC application and compare it against an AMD-validated Full-ELF Windows test.

**Recommended medium-term action:** validate persistent XCLBIN/context execution and dynamic instruction-BO updates as an independent architecture, without waiting for the native Full-ELF problem to be resolved.

**Recommended longer-term action:** integrate correctness, stability, and performance regression tests into Windows CI using matched Linux/Windows workloads wherever practical.

The current evidence supports pursuing these paths in parallel. It does not support attributing all observed failures or performance deficits to a single XRT, firmware, or compiler defect.

**Final technical verdict:** Multiple confirmed implementation and compatibility problems, meaningful architectural interconnections, no proven universal root cause, and a concrete validation path suitable for joint community/AMD investigation.
