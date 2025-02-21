import pickle
import argparse
import os

def get_results_root(args):
    """Construct the root path for the results based on the provided arguments.
    
    Args:
        args (argparse.Namespace): The command-line arguments.
    
    Returns:
        str: The constructed root path for the results.
    """
    return os.path.join(
        args.root,
        args.dataset,
        args.model,
        args.task
    ) 

def get_path(args, no_perturbations=False):
    """Construct the file path for the results based on the provided arguments.
    
    Args:
        args (argparse.Namespace): The command-line arguments.
        no_perturbations (bool, optional): Flag to indicate if the path should
                                           be for no perturbations.
    
    Returns:
        str: The constructed file path for the results.
    """
    filename = "none-results.pkl" if no_perturbations else f"{args.perturbation}-results.pkl"
    results_root = get_results_root(args)
    return os.path.join(results_root, filename)

def get_rsum(dataf):
    """Calculate the sum of recall scores at 1, 5, and 10.
    
    Args:
        dataf (pd.DataFrame): The dataframe containing the results.
    
    Returns:
        float: The sum of recall scores at 1, 5, and 10.
    """
    return dataf['t2i_recalls_at_1'] + dataf['t2i_recalls_at_5'] + dataf['t2i_recalls_at_10']

def main(args):
    """Main function to process and print the results based on the specified conditions.
    
    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    none_path = get_path(args, no_perturbations=True)
    with open(none_path, 'rb') as f:
        dataset_no_perturbation = pickle.load(f)

    perturbations_root = get_results_root(args)
    perturbation_files = os.listdir(perturbations_root)

    filtered_perturbation_files = [
        file for file in perturbation_files
        if 'none' not in file and 'DS_Store' not in file and not os.path.isdir(file) and 'splits' not in file
    ]

    for perturbation_file in filtered_perturbation_files:
        perturbation_path = os.path.join(perturbations_root, perturbation_file)
        print('perturbation_file: ', perturbation_file)

        with open(perturbation_path, 'rb') as f:
            dataf_perturbation = pickle.load(f)

        dataset_no_perturbation['t2i_rsum'] = get_rsum(dataf=dataset_no_perturbation)
        dataf_perturbation['t2i_rsum'] = get_rsum(dataf=dataf_perturbation)
        dataf_perturbation['original_query'] = dataset_no_perturbation['t2i_queries']

        merged_df = dataset_no_perturbation.merge(
            dataf_perturbation,
            left_on='t2i_queries',
            right_on='original_query',
            suffixes=['_none', '_perturbation']
        )

        merged_df['rsum_diff'] = merged_df['t2i_rsum_none'] - merged_df['t2i_rsum_perturbation']

        len_df = merged_df.shape[0]
        df_rsum_increased = merged_df[merged_df['rsum_diff'] < 0]
        df_rsum_decreased = merged_df[merged_df['rsum_diff'] > 0]
        df_rsum_unchanged = merged_df[merged_df['rsum_diff'] == 0]

        print(
            f"""
            Statistics:
            model: {args.model}
            dataset: {args.dataset}
            Perturbation: {perturbation_path.split('/')[-1]}
            Perturbation [decreases]/[increases]/[doesn't affect] the rsum in the following cases
            {round(df_rsum_decreased.shape[0]*100/len_df, 2)}\t{round(df_rsum_increased.shape[0]*100/len_df, 2)}\t{round(df_rsum_unchanged.shape[0]*100/len_df, 2)}
            """
        )

        results_root = os.path.join(perturbations_root, 'splits')
        print("results_root", results_root)
        os.makedirs(results_root, exist_ok=True)
        perturbation_name = perturbation_file.split('-')[0]
        file_rsum_decreased = os.path.join(results_root, f'{perturbation_name}-rsum-decreased.pkl')
        file_rsum_increased = os.path.join(results_root, f'{perturbation_name}-rsum-increased.pkl')
        file_rsum_unchanged = os.path.join(results_root, f'{perturbation_name}-rsum-unchanged.pkl')

        with open(file_rsum_decreased, 'wb') as f:
            pickle.dump(df_rsum_decreased, f)
        
        with open(file_rsum_increased, 'wb') as f:
            pickle.dump(df_rsum_increased, f)
        
        with open(file_rsum_unchanged, 'wb') as f:
            pickle.dump(df_rsum_unchanged, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["f30k", "f30k_aug", "coco", "coco_aug"],
        help="dataset name: f30k, f30k_aug, coco, coco_aug",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="align",
        choices=["clip", "align", "altclip", "groupvit"],
        help="model name: clip, align, altclip, groupvit",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2i",
        choices=["t2i", "i2t"],
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        default="char_swap",
        choices=[
            "none",
            'char_swap',
            'missing_char',
            'extra_char',
            'nearby_char',
            'probability_based_letter_change',
            "synonym_noun",
            'synonym_adj',
            "distraction_true",
            "distraction_false",
            "shuffle_nouns_and_adj",
            'shuffle_all_words',
            'shuffle_allbut_nouns_and_adj',
            'shuffle_within_trigrams',
            'shuffle_trigrams'
        ],
        help="perturbation type: none, typos, etc.",
    )
    args = parser.parse_args()
    main(args)
