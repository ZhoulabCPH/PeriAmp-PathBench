# VPC

Research code for whole-slide image tiling, pathology encoder fine-tuning,
multiple-instance learning (MIL), ensemble evaluation, and model interpretation.

## Project layout

| Directory | Purpose |
| --- | --- |
| Tiling | Extract patches from whole-slide images (WSIs), match tiles to GeoJSON annotations, and migrate legacy tile coordinates. |
| Finetune/ExPLoRA-main | Fine-tune the GigaPath and Virchow2 tile encoders and export merged LoRA weights. |
| MIL_pipeline | Run MIL cross-validation and external evaluation on slide-level HDF5 feature bags. |
| Ensemble learning | Combine classifier predictions and generate evaluation reports. |
| interpreability | Analyze tile attention and integrated gradients, cluster features, and render slide overlays. |

## Dependencies

- Python 3.10
- `torch`, `torchvision`, `monai`
- `numpy`, `scipy`, `pandas`, `h5py`, `openpyxl`, `scikit-learn`, `joblib`
- `openslide-python`, OpenSlide (`openslide-bin`), `pillow`, `scikit-image`, `shapely`, `opencv-python`, `imageio`
- `matplotlib`, `seaborn`, `tqdm`, `progressbar2`, `psutil`, `tensorboard`
- GigaPath and Virchow2 — `wilds`, `tifffile`, `safetensors`
- Distributed fine-tuning — Linux, CUDA, NCCL
- TransMIL — `nystrom-attention`
- WiKG — `torch-geometric`
- Slide overlays — `pikepdf`, `ijson`, optional `tiffslide`
- UMAP — `umap-learn`
- Leiden clustering — `scanpy`, `igraph`, `leidenalg`
- WebDataset and YAML configurations — `webdataset`, `pyyaml`
- Bundled ExPLoRA examples — `omegaconf`, `fvcore`, `wandb`, `torchmetrics`, `xformers`

## Foundation model weights

The benchmark uses 12 pathology foundation models as frozen feature extractors, plus an ImageNet-pretrained ResNet-50 baseline.

| Model | Model Resources |
| --- | --- |
| CTransPath | https://github.com/Xiyue-Wang/TransPath |
| UNI | https://huggingface.co/MahmoodLab/UNI |
| UNI2-h | https://huggingface.co/MahmoodLab/UNI2-h |
| Prov-GigaPath | https://huggingface.co/prov-gigapath/prov-gigapath |
| Phikon-v2 | https://huggingface.co/owkin/phikon-v2 |
| Virchow2 | https://huggingface.co/paige-ai/Virchow2 |
| CHIEF | https://github.com/hms-dbmi/CHIEF |
| H-optimus-0 | https://huggingface.co/bioptimus/H-optimus-0 |
| H-optimus-1 | https://huggingface.co/bioptimus/H-optimus-1 |
| CONCH | https://huggingface.co/MahmoodLab/CONCH |
| TITAN | https://huggingface.co/MahmoodLab/TITAN |
| MUSK | https://huggingface.co/xiangjx/musk |
| ResNet-50 | https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html |


## Third-party code

| Component |  Source |
| --- | --- |
| Prov-GigaPath (tiling code) | https://github.com/prov-gigapath/prov-gigapath |
| ExPLoRA | https://github.com/samar-khanna/ExPLoRA |
| DINOv2 | https://github.com/facebookresearch/dinov2 |
| CLAM | https://github.com/mahmoodlab/CLAM |
| TransMIL | https://github.com/szc19990412/TransMIL |
| WiKG | https://github.com/WonderLandxD/WiKG |
| CellViT++ | https://github.com/TIO-IKIM/CellViT-plus-plus |



## Local inputs and outputs

Run the tiling, MIL, ensemble, and interpretation commands from the repository
root. Put local inputs under data/ and model checkpoints under weights/.
Use the path options shown by a script's --help to provide a different location.

- Place WSIs under data/slides/. The annotated tiling script can read paired
  GeoJSON annotations from a separate directory using --geojson-dir.
- Place feature bags and label workbooks under data/features/ and data/labels/.
  The MIL loader expects HDF5 datasets named features and coords. The label
  workbook uses ID, available, and one column per class. Each ID must match
  the corresponding HDF5 file stem.
- The MIL script lists its feature directories and label files in
  MIL_pipeline/cross_validation_All.py. The default cohort names are internal,
  external_a, and external_b. The configured external paths and workbooks use
  the same order.

## Example entry points

Inspect the arguments for tiling and interpretation from the repository root.

```sh
python Tiling/openslide_mdsx.py --help
python Tiling/get_tile_standalone_geojson.py --help
python interpreability/heatmap/calculate_attention_batch.py --help
```

Run the ExPLoRA commands from Finetune/ExPLoRA-main so the relative paths in
the JSON configurations resolve from that directory. Copy an example to a
local configuration file ending in .local.json and adjust its input and output
paths. 

```sh
cd Finetune/ExPLoRA-main
python -m gigapath.train_gigapath_ddp --help
python -m virchow2.train_virchow2_ddp --help
python export_merge_lora_gigapath.py --help
python export_merge_lora_virchow2.py --help
```

## License

VPC's original code is available under the [MIT License](LICENSE).

Third-party code and pretrained model weights are subject to the terms
published by their respective providers. See the upstream sources for
[third-party code](#third-party-code) and [model weights](#foundation-model-weights).

## Citation

If you use this software in your research, please cite VPC. Machine-readable
citation metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@misc{vpc,
  author = {{VPC contributors}},
  title  = {VPC},
  note   = {Research software for whole-slide image analysis}
}
```
