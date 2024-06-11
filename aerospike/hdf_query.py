#!/usr/bin/env python3
import asyncio
import argparse

from logging import _nameToLevel as LogLevels
from aerospikehdf import Aerospike, OperationActions as Actions

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    Aerospike.parse_arguments_query(parser)
    
    args = parser.parse_args()
    
    return args

async def main_loop() -> None:
    
    with Aerospike(args, Actions.QUERY) as asInstance:    
        await asInstance.get_dataset()
        await asInstance.query()
    
if __name__ == "__main__":
    args = parse_arguments()
    
    asyncio.run(main_loop())