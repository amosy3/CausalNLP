Create Concept Activation Vectors (CAVs):
'''
python main.py --use_gpu=1 --method=tcav --log_file='test_logs'
'''

Run ConceptShap using the CAVs:
'''
python main.py --use_gpu=1 --method=ConceptShap --log_file='test_logs'
'''
