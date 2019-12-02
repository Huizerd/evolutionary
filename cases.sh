# First see whether 400 generations is enough, and how these baselines do
# And test how to randomize env: for each height for each individual, for each height for each generation, or for each generation?
python main.py --config configs/case0.yaml --tags case0 allrand --verbose 1
python main.py --config configs/case0.yaml --tags case0 allrand --verbose 1
#python main.py --config configs/case0.yaml --tags case0 --verbose 1
#python main.py --config configs/case0.yaml --tags case0 --verbose 1
python main.py --config configs/case1.yaml --tags case1 allrand --verbose 1
python main.py --config configs/case1.yaml --tags case1 allrand --verbose 1
#python main.py --config configs/case1.yaml --tags case1 --verbose 1
#python main.py --config configs/case1.yaml --tags case1 --verbose 1
