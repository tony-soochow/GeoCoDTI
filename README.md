# GeoCoDTI

Implementation of the manuscript **"Confidence Meets Geometry: Geometry-aware Collaborative Learning for Few-shot Multimodal Drug-Target Interaction Prediction"**. GeoCoDTI combines intrinsic and extrinsic experts and uses prediction confidence together with prototype-based geometric uncertainty to select pseudo labels.

## Usage

1. Download [DRKG triples](https://drive.google.com/file/d/1-pRYRtgcNFqxeL3Q9ZgxIU5rnJbfKB8M/view?usp=sharing) and place `drkg.tsv` in `var_data/`. Download the [pretrained KGE model](https://drive.google.com/file/d/1_RCRrHJBosWycpqzxXxzrmJYCGWaJfc8/view?usp=sharing) and place its `.pth` file in `var_models/`. Also provide `entities.dict`, `relations.dict`, `ent_id2seq.csv`, and `ent_id2smiles.csv` in `var_data/`, and the ESM-1b checkpoint at `esm/esm1b_t33_650M_UR50S.pt`. Install a compatible PyTorch and DGL setup for your machine, followed by the packages in `requirements.txt`.

2. Prepare the few-shot train, validation, and test splits under `var_data/<dataset>/`. Each split requires `train.tsv`, `train_neg.tsv`, `valid.tsv`, `valid_neg.tsv`, `test.tsv`, and `test_neg.tsv`. The current `split_data.py` only creates output directories; it does not generate these splits.

3. From the repository root, train and evaluate GeoCoDTI with the pretrained KGE model:

   ```bash
   python kge/std_main.py --dataset ago_10shots_0 --device 0 --load_kge_model 2024-04-28_10_01_40.24__kgeSLHstd_main.py--save--dataset__a-10--device__6--gate__kge.pth --pseudo_pos_threshold 0.7 --pseudo_neg_threshold 0.3 --pseudo_uncertainty_max 0.25
   ```

   Replace the dataset or checkpoint name as needed. The ESM path can be changed with `--esm_model_path`. Omitting `--load_kge_model` trains KGE from `var_data/drkg.tsv`. Use `--save` to save model components and the corresponding `--load_*` arguments to reload them.