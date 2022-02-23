# gpu-link
A tool for monitoring and displaying GPU stats

## Installation
1. Create pipenv environment: `pipenv install`  
2. Run recording `python gpu-link.py`
3. Stop recording *Ctrl+C*
4. *mem_consumption.png* is created in the current working directory

## Example Output
```
GPU[0]: min memory consumption: 2234.70703125[MB]  
GPU[0]: max memory consumption: 1808.6171875[MB]
```
![Memory consumption over time](./docs/mem_consumption.png)

