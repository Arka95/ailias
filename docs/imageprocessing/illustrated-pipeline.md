[Back to REPO_ROOT](../../README.md) | [Imageprocessing](README.md)

# Photo to facial features

**Actual images below; isolated demonstrations, not one completed production run.**

**New API:** `ImageProcessor.process_image()` prepares **original -> MODNet ->
BiSeNet -> optional white balance**. Then `FaceFeatureExtractor.extract_features()`
runs **MediaPipe -> features**, reusing the existing head, original RGB and region
labels. No repeated head isolation, YuNet or alignment. [Usage](README.md#simple-python-api).

## Existing CLI: step-by-step image flow

| Step | Actual image | What passes forward |
|---|---|---|
| **1. Original photo** | ![Original](images/original.png) | RGB pixels |
| **↓ 2. YuNet** | ![YuNet face box and five points](images/yunet.png) | Face box + five points. SFace separately computes an identity embedding. |
| **↓ 3. Expand box and crop** | ![Expanded crop](images/expanded-crop.png) | Native-resolution crop + original-image bounds. This preview reproduces the recorded crop coordinates. |
| **↓ 4a. MODNet** | ![Actual YuNet-crop-to-MODNet result](images/modnet.png) | Soft transparency. Neck/shirt remain. |
| **4b. BiSeNet, alongside MODNet** | ![Direct BiSeNet example](images/bisenet-only.png) | Head/skin/hair/etc. labels. **Shown: separate full-image test, not the crop above.** |
| **↓ 5. Combine opacity + head labels** | ![MODNet plus BiSeNet head example](images/modnet-bisenet.png) | Head-only RGBA. **Shown: full-image MODNet → grey composite → BiSeNet experiment; not the production merge.** |
| **↓ 6. White balance / exposure** | **Not demonstrated** | Corrected RGB when evidence supports it; otherwise skipped. No additional CNN. |
| **↓ 7. Align using YuNet's eye points** | **Not demonstrated** | Separate 512 × 512 analysis image + reversible coordinate transform |
| **↓ 8. MediaPipe** | **Not demonstrated** | RGB image → 478 facial points + expression scores; points mapped back to original |
| **↓ 9. Extract features** | **Not demonstrated** | Original RGB + dense points + fresh BiSeNet labels → facial ratios, skin/lip/brow/hair colours, iris evidence and review flags |

**Production distinction:** MODNet and BiSeNet both receive the RGB crop; their outputs are combined. BiSeNet does **not** receive MODNet's grey composite in production. Feature colours are sampled from original pixels, not the corrected cutout.

## Existing CLI architecture

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 850}}}%%
flowchart TB
    subgraph MAIN[" "]
        direction TB
        A["1. Decode original RGB"]:::svc --> B["2. YuNet detection and SFace embeddings"]:::svc
        B --> C["3. Expanded crop, grouping and review"]:::svc
        C --> D["4a. MODNet: soft opacity"]:::svc
        C --> E["4b. BiSeNet: region labels"]:::svc
        D --> F["5. Combine alpha and head labels"]:::svc
        E --> F
        F --> Q["6. Colour correction"]:::svc
        A -->|"scene-neutral colour estimate"| Q
        Q --> G["7. Head cutout and five-point alignment"]:::svc
        G --> H["8. MediaPipe on aligned RGB; map points to source"]:::svc
        H --> J["9. Features from original RGB, dense points and BiSeNet labels"]:::svc
        A -->|"original pixels"| J
        C -->|"portrait bounds"| J
        J --> K["Per-photo measurements, colours and review status"]:::store
    end
    MAIN ~~~ SUM
    SUM["Existing prepared-photo route. Read top to bottom; blue is processing, green is output. MODNet and BiSeNet process RGB independently; feature extraction reparses original RGB. Backed by: pipeline.py, cleansing.py, dense.py and feature_evidence.py."]:::caption
    style MAIN fill:none,stroke:none
    classDef svc fill:#cfe3ff,stroke:#2b6cb0,color:#0b2545
    classDef store fill:#d6f5df,stroke:#2f855a,color:#12351f
    classDef caption fill:#f7f7f7,stroke:#999,color:#222,font-size:16px
```

**Backed by:** [scan/crop](../../src/ailias/imageprocessing/pipeline.py#L134-L175) · [merge/colour/alignment](../../src/ailias/imageprocessing/cleansing.py#L253-L295) · [MediaPipe](../../src/ailias/imageprocessing/dense.py#L35-L78) · [final features](../../src/ailias/imageprocessing/feature_evidence.py#L95-L125).

Image previews are included in the adjacent `images` folder. Missing stages were not run for this guide.
