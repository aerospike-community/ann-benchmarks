#!/usr/bin/env python3
import asyncio
import argparse

from aerospikedataset import AerospikeDS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    AerospikeDS.parse_arguments(parser)
    
    args = parser.parse_args()
    
    return args

async def main_loop(args : argparse.Namespace) -> None:
    
    async with AerospikeDS(args) as asInstance:    
        await asInstance.populate_vector_info()
        await asInstance.Generate_hdf_dataset()
        
if __name__ == "__main__":
    args = parse_arguments()
    
    asyncio.run(main_loop(args))


