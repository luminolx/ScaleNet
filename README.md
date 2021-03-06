# ScaleNet
ScaleNet: Searching for the Model to Scale (ECCV 2022) [ECVA](https://XXX "ECVA") or [ArXiv](https://XXX "ArXiv")

## Requirements
- python >= 3.6
- 1.0.0 <= PyTorch <= 1.3.0
- torchvision >= 0.3.0

## Super-supernet Training
- Download datasets
- Run: `bash train.sh`

## Searching
- Run: `bash search.sh`

## Retraining
- Please directly apply the package [TIMM](https://github.com/rwightman/pytorch-image-models "TIMM").

## Warning
- dataloader and dataset may need to be modified for adapting your environment.

## Citation
If you find this paper useful in your research, please consider citing:
```
@InProceedings{xie2022scalenet,
  author={Jiyang Xie and Xiu Su and Shan You and Zhanyu Ma and Fei Wang and Chen Qian},
  booktitle={European Conference on Computer Vision (ECCV)}, 
  title={{ScaleNet}: {S}earching for the Model to Scale}, 
  year={2022},
  volume={30},
}
```
