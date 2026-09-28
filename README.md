# NV-Reason-CT in the browser

**https://jarrelscy.github.io/nv-reason-ct-web/**

Runs [NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT), NVIDIA's 3D CT vision-language model, entirely in the browser with WebGPU. Drop in a chest or abdominal CT and it writes a structured report, gives a reasoning analysis, or answers follow-up questions about the scan.

Images never leave the computer. They are read and processed in the browser tab and nothing is uploaded. The model files are downloaded from [Hugging Face](https://huggingface.co/jarrelscy/nv-reason-ct-onnx) on first use and cached in the browser: 5.0 GB for the 8-bit build or 3.0 GB for the 4-bit build.

> **Research use only.** This is not a medical device. It has not been validated or approved for clinical use and must not be used for diagnosis or treatment decisions. The model can miss findings and can describe findings that are not there.

## Inputs

- a zip of a DICOM study, a DICOM folder or loose DICOM files
- a `.nii` / `.nii.gz` volume

Choose Chest or Abdomen before loading the scan. From a study, the largest axial CT series whose Series Description or Body Part Examined matches the region is used ("chest", "thorax", "lung" for chest, "abd" for abdomen), and otherwise the largest CT series. Supported DICOM transfer syntaxes are uncompressed, deflated, RLE, JPEG lossless, JPEG-LS, JPEG 2000 and HTJ2K.

## Requirements

- Recent desktop Chrome or Edge with WebGPU. Firefox and Safari are untested.
- A GPU with at least 8 GB of memory for the 8-bit build (12 GB or more recommended), or 6 GB for the 4-bit build.
- 16 GB RAM; the tab peaks at about 6 GB while the models load.
- 6 GB of browser storage for the 8-bit build (3.5 GB for 4-bit). Internet is needed on the first visit only.

On an RTX PRO 6000 (Linux, Vulkan, no shader-f16) the vision encoder takes 2 s, reading the 13,845-token prompt takes about 15 s, and generation runs at 45–50 tokens/s with either build, so a report takes about 25 s once the models are cached. Follow-up questions reuse the cached prompt state and start straight away.

## How it works

1. **Preprocessing** (`lib/preprocess.js`) follows `ImageLoader3D` in the model's `processor.py`: orient to LPS, resample to 2 mm, pad to 192 with -1000 HU, crop a 192³ cube around the chest or abdomen using a lung mask to place it in z, then clip to [-1000, 1000] HU and scale to [-1, 1]. It matches the PyTorch/MONAI crop on the test cases.
2. **Vision encoder** (`vision.onnx`, 184 MB): the 147M-parameter Primus 3D ViT and merger turn the crop into 13,824 tokens of width 2560. Attention over 13,824 tokens is split into query chunks so no score matrix is larger than about 760 MB. Weights are int8 (row cosine similarity with fp32 ≥ 0.99998).
3. **Language model** (`int8/decoder.onnx` or `int4/decoder.onnx`): the Qwen3.5-4B hybrid decoder (24 Gated DeltaNet layers and 8 full-attention layers), exported with the onnxruntime-genai builder with fp32 activations and the lm_head pruned to the last token. The token embeddings are tied to lm_head, so they are not shipped twice: `lib/nvreason.js` dequantises each token's row from the lm_head weights in the cached files. The image tokens get 3D MRoPE positions as in the original model. The prompt is prefilled in chunks of 1024 tokens and the recurrent and attention state stays on the GPU between turns.

### Quantisation

Teacher-forced agreement with the fp32 model on the reference reports of two test cases (top-1 token agreement and mean KL divergence of the next-token distribution):

| build | download | case 1 | case 2 |
|---|---|---|---|
| int8 (RTN, block 32) | 5.0 GB | 99.1%, KL 0.0002 | 99.2%, KL 0.0004 |
| int4 GPTQ (block 32, act order) | 3.0 GB | 97.8%, KL 0.020 | 97.7%, KL 0.066 |
| int4 round-to-nearest (not shipped) | 3.0 GB | 97.2%, KL 0.047 | 96.7%, KL 0.10 |

With greedy decoding the 8-bit build in the browser produces the same report as the fp32 PyTorch model on the test case. The 4-bit builds write well-formed reports but more often drop or change findings; on the first test case the 4-bit GPTQ build calls the pancreas atrophic where the original model reports calcifications of chronic pancreatitis. GPTQ calibration uses the model's own reports and reasoning analyses on six crops from four other scans (`tools/quant.py`).

## Running locally

Any static file server works; `coi-serviceworker.js` enables cross-origin isolation:

```
python3 -m http.server 8000   # then open http://localhost:8000/
```

`?models=models/hub/` loads model files from a local folder laid out like the Hugging Face repo. `tools/` has the export, quantisation, packaging and test scripts.

## Licence and attribution

The code in this repository is MIT licensed (`LICENSE`). NV-Reason-CT and the ONNX files derived from it are by NVIDIA and distributed under the OpenMDW License 1.1 (`LICENSE-OpenMDW`). Bundled third-party libraries are listed in `THIRD_PARTY_LICENSES.md`.
