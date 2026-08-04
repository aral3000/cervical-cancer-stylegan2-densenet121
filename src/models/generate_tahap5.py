import json
import traceback

source_notebook = '8-sempro-tahap4-6k-8k_optimized.ipynb'
target_notebook = '8-sempro-tahap5-8k-10k.ipynb'

try:
    with open(source_notebook, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            source_list = c.get('source', [])
            new_source = []
            modified = False
            
            for line in source_list:
                if line.strip().startswith('RESTORE_DIR ='):
                    # Replace with the new path for Tahap 5
                    new_source.append("RESTORE_DIR = '/kaggle/input/hasil-tahap-4-anda/training_checkpoints'\n")
                    modified = True
                else:
                    new_source.append(line)
                    
            if modified:
                c['source'] = new_source
                break
                
    with open(target_notebook, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f'Done generating {target_notebook} securely!')
    
except Exception as e:
    print('error:', e)
    traceback.print_exc()
