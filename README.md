Create Concept Activation Vectors (CAVs):
'''
python main.py --method=tcav --dataset=cv --backbone=gpt2 --log_file=full  
'''

Run ConceptShap using the CAVs:
'''
python main.py --use_gpu=1 --method=ConceptShap --dataset=cv --backbone=gpt2 --load_from_dir='full_2025-06-11_10-41-23'
'''
