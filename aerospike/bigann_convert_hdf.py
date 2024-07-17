#!/usr/bin/env python3

import os
import argparse
import asyncio
from bigann.datasets import DATASETS, DatasetCompetitionFormat
from bigann.bigann_convert import BigAnnConvert

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    
    BigAnnConvert.parse_arguments(parser)
    
    args = parser.parse_args()
    
    return args

async def main_loop(args : argparse.Namespace) -> None:
    
    ds : DatasetCompetitionFormat = DATASETS[args.dataset]()
    
    if not os.path.exists(ds.basedir):
        raise FileNotFoundError(f"Big ANN Folder '{ds.basedir}' was not found for dataset '{args.dataset}' ({ds.short_name()}). Do you forget to download the dataset? Try running 'bigann_download.py --dataset {args.dataset}'...")
    
    async with BigAnnConvert(args, ds) as convertInstance:
        await convertInstance.bigann_getinfo()
    

if __name__ == "__main__":
    args = parse_arguments()
    
    asyncio.run(main_loop(args))
    