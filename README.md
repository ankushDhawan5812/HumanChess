# HumanChess

Because the actual LiChess and ChessBench datasets are too large (would cause git LFS) those were not included in the repo. Additionally, the model weights are included but the paths in all the code files would need to be updated to properly run the training scripts. The key training scripts are sv_train.py and diffuse_trainer.py. Additionally, there were some odd tweaks here and there since we were running the training for days on end on a local 4090, hence the hardcoded file paths.
