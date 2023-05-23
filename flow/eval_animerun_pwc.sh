python -u evaluate_pwc.py --model checkpoints/20000_pwc-ft-sintel.pth --dataset anime --mixed_precision
python -u evaluate_pwc.py --model checkpoints/20000_pwc-ft-creative.pth --dataset anime --mixed_precision
python -u evaluate_pwc.py --model checkpoints/20000_pwc-animerun-v2-ft.pth --dataset anime --mixed_precision

