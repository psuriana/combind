import os
from schrodinger.structure import StructureReader
from subprocess import run
import shutil

command = '$SCHRODINGER/utilities/prepwizard -WAIT -rehtreat -watdist 0 {}_in.mae {}_out.mae'

def load_complex(prot_in, lig_in, struct):

    prot_st = next(StructureReader(prot_in))

    if not os.path.exists(lig_in):
        prot_st.title = struct
        return prot_st

    lig_st = next(StructureReader(lig_in))

    assert len(lig_st.chain) == 1, struct
    for c in lig_st.chain:
        c.name = 'L'

    alpha = 'ABCDEFGHIJKMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    alpha_count = 0
    for c in prot_st.chain:
        if c.name.strip() == '': continue
        try:
            c.name = alpha[alpha_count]
        except Exception as e:
            print(alpha_count)
            raise e
        alpha_count += 1

    merged_st = lig_st.merge(prot_st)
    merged_st.title = struct
    return merged_st

def struct_process(structs,
                   protein_in='structures/raw/{pdb}_prot.mae',
                   ligand_in='structures/raw/{pdb}_lig.mae',
                   processed_in='structures/processed/{pdb}/{pdb}_in.mae',
                   processed_out='structures/processed/{pdb}/{pdb}_out.mae',
                   processed_sh='structures/processed/{pdb}/process.sh'):

    for struct in structs:
        _protein_in = protein_in.format(pdb=struct)
        _ligand_in = ligand_in.format(pdb=struct)
        _processed_in = processed_in.format(pdb=struct)
        _processed_out = processed_out.format(pdb=struct)
        _processed_sh = processed_sh.format(pdb=struct)
        _workdir = os.path.dirname(_processed_sh)

        if os.path.exists(_processed_out):
            continue

        print('processing', struct)

        os.system('mkdir -p {}'.format(os.path.dirname(_workdir)))
        os.system('rm -rf {}'.format(_workdir))
        os.system('mkdir {}'.format(_workdir))

        shutil.copy(_protein_in, _processed_in)

        # prep AF structure without the ligand otherwise it may bias the sidechain to crystal
        with open('{}/process_in.sh'.format(_workdir), 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(command.format(struct, struct))
        run('sh process_in.sh', shell=True, cwd=_workdir)

        merged_st = load_complex(_processed_out, _ligand_in, struct)
        merged_st.write(_processed_out)
