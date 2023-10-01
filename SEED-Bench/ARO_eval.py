from evaluator.evaluation_ARO import BLIP2HFModelWrapper
import argparse
import os
import json
import pandas as pd

def main():
    evaluator = BLIP2HFModelWrapper(root_dir = "./data" , device="cuda")
    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench/Statements.json')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    qa_anno = json.load(open(args.anno_path, 'rb'))
    if 'questions' in qa_anno.keys():
        qa_anno = qa_anno['questions']

    x = pd.DataFrame(qa_anno)
    df = x[x.question_type_id == 1]
    y = df.to_dict(orient="records")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    evaluator.get_retrieval_scores_batched(joint_loader=y)



    # The interface for testing MLLMs



if __name__ == '__main__':
    main()