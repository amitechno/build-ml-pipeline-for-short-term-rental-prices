#!/usr/bin/env python
"""
Download raw data from W&B, do the basic cleaning and upload the artifact back to W&B
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    
    #Download the artifact from Wandb
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    
    #Drop the outliers
    logger.info("Dropping outliers")
    df = pd.read_csv(artifact_local_path )
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    df.to_csv("clean_sample.csv", index=False)
    
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
 )
    artifact.add_file("clean_sample.csv")
    logger.info("Logging artifact")
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=int,
        help="minimum threshold value for price of airbnb",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=int,
        help="maximum threshold value for price of airbnb",
        required=True
    )


    args = parser.parse_args()

    go(args)
