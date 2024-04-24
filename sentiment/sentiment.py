import argparse
from transformers import pipeline
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser("Sentiment Analysis")
    parser.add_argument('comments_cleaned', help='file with the text to analysis on CSV format')
    parser.add_argument('comment_text', help='column in the file that contains the text')
    args = parser.parse_args()

    df_text = pd.read_csv(args.file)#, lineterminator='\n')
    # model = 'finiteautomata/bertweet-base-sentiment-analysis'
    model = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
    sentiment = pipeline(task='sentiment-analysis', model=model)
    counter = 0

    # Define the total number of iterations
    total_iterations = len(df_text)
    
    # Create a progress bar
    progress_bar = tqdm(total=total_iterations, position=0, leave=True)

    for index, row in df_text.iterrows():
        try:
            df_text.loc[index, 'sentiment'] = str(sentiment(str(row[args.text_column])[:279])[0]['label'])
            df_text.loc[index, 'sentiment_score'] = float(sentiment(str(row[args.text_column])[:279])[0]['score'])
        except Exception as e:
            print(f'{str(row[args.text_column])}', f'Error -> {e}')
        counter += 1
        progress_bar.update(1)  # Update the progress bar
        progress_bar.set_description(f'Processed: {counter}/{total_iterations}')

    df_text.to_csv(f'sentiment_analysis_{args.file}', index=False)


if __name__ == '__main__':
    main()