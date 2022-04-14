import os
import numpy as np
from glob import glob
from schrodinger.structure import StructureReader
from utils import basename, mp, mkdir, np_load

IFP = {'rd1':    {'version'           : 'rd1',
                   'level'             : 'residue',
                   'hbond_dist_opt'    : 2.5,
                   'hbond_dist_cut'    : 3.0,
                   'hbond_angle_opt'   : 60.0,
                   'hbond_angle_cut'   : 90.0,
                   'sb_dist_opt'       : 4.0,
                   'sb_dist_cut'       : 5.0,
                   'contact_scale_opt' : 1.25,
                   'contact_scale_cut' : 1.75,
                   'pipi_dist_cut'     : 8.0,
                   'pipi_dist_opt'     : 7.0,
                   'pipi_norm_norm_angle_cut'     : 30.0,
                   'pipi_norm_centroid_angle_cut' : 45.0,
                   'pipi_t_dist_cut': 6.0,
                   'pipi_t_dist_opt': 5.0,
                   'pipi_t_norm_norm_angle_cut': 60.0,
                   'pipi_t_norm_centroid_angle_cut': 45.0},
      }

class Features:
    """
    Organize feature computation and loading.
    """
    def __init__(self, root, ifp_version='rd1', shape_version='pharm_max',
                 mcss_version='mcss16', max_poses=10000, pv_root=None,
                 ifp_features=['hbond', 'saltbridge', 'contact', 'pipi', 'pi-t']):
        self.root = os.path.abspath(root)
        if pv_root is None:
            self.pv_root = self.root + '/docking'

        self.ifp_version = ifp_version
        self.shape_version = shape_version
        self.mcss_version = mcss_version
        self.mcss_file = '{}/features/{}.typ'.format(os.environ['COMBINDHOME'], mcss_version)
        self.max_poses = max_poses
        self.ifp_features = ifp_features

        self.raw = {}

    def path(self, name, base=False, pv=None, pv2=None):
        if base:
            return '{}/{}'.format(self.root, name)

        if self.pv_root != self.root+'/docking':
            if pv1 is not None:
                pv = pv.replace(self.pv_root), self.root+'/single'
            if pv2 is not None:
                pv2 = pv2.replace(self.pv_root), self.root+'/single'

        # single features
        if name == 'rmsd':
            return pv.replace('_pv.maegz', '_rmsd.npy')
        elif name == 'gscore':
            return pv.replace('_pv.maegz', '_gscore.npy')
        elif name == 'all_gscores':
            return pv.replace('_pv.maegz', '_all_gscores.npy')
        elif name == 'name':
            return pv.replace('_pv.maegz', '_name.npy')
        elif name == 'ifp':
            suffix = '_ifp_{}.csv'.format(self.ifp_version)
            return pv.replace('_pv.maegz', suffix)

        # pair features
        elif name == 'shape':
            return '{}/shape/shape-{}-and-{}.npy'.format(self.root,
                basename(pv), basename(pv2))
        elif name == 'mcss':
            return '{}/mcss/mcss-{}-and-{}.npy'.format(self.root,
                basename(pv), basename(pv2))
        else:
            ifp1 = basename(self.path('ifp', pv=pv))
            ifp2 = basename(self.path('ifp', pv=pv2))
            return '{}/ifp-pair/{}-{}-and-{}.npy'.format(self.root, name, ifp1, ifp2)

    def get_poseviewers(self):
        return glob(self.pv_root+'/*/*_pv.maegz')

    def load_features(self, pvs=None, delete=False,
                      features=['shape','mcss','hbond','saltbridge','contact']):
        if pvs is None:
            pvs = self.get_poseviewers()

        self.raw = {}

        self.raw['rmsd'] = {}
        for pv in pvs:
            name = basename(pv)
            path = self.path('rmsd', pv=pv)
            if os.path.exists(path):
                self.raw['rmsd'][name] = np_load(path, delete=delete, halt=not delete)

        self.raw['gscore'] = {}
        for pv in pvs:
            name = basename(pv)
            path = self.path('gscore', pv=pv)
            self.raw['gscore'][name] = np_load(path, delete=delete, halt=not delete)

        for feature in features:
            self.raw[feature] = {}
            for i, pv1 in enumerate(pvs):
                for pv2 in pvs[i+1:]:
                    path = self.path(feature, pv=pv1, pv2=pv2)
                    name1 = basename(pv1)
                    name2 = basename(pv2)
                    self.raw[feature][(name1, name2)] = np_load(path, delete=delete, halt=not delete)

    def is_single_complete(self, pvs, ifp=True):
        class IsDone:
            def __init__(self):
                self.is_done = True

            def __call__(self, f, x, p):
                if len(x):
                    self.is_done = False

        is_done = IsDone()
        self.compute_single_features(pvs, ifp=ifp, run=is_done)
        return is_done.is_done

    def is_pair_complete(self, pvs, ifp=True, shape=True, mcss=True, all_energy=False):
        class IsDone:
            def __init__(self):
                self.is_done = True

            def __call__(self, f, x, p):
                if len(x):
                    self.is_done = False

        is_done = IsDone()
        self.compute_pair_features(pvs, ifp=ifp, shape=shape, mcss=mcss, run=is_done,
                                   all_energy=all_energy)
        return is_done.is_done

    def compute_single_features(self, pvs, ifp=True, processes=1, run=mp, all_energy=False,
                                overwrite=False):
        # For single features, there is no need to keep sub-sets of ligands
        # seperated,  so just merge them at the outset to simplify the rest of
        # the method.
        if type(pvs[0]) == list:
            pvs = [pv for _pvs in pvs for pv in _pvs]

        pvs = [os.path.abspath(pv) for pv in pvs]

        print('Extracting glide scores.')
        for pv in pvs:
            out = self.path('gscore', pv=pv)
            if overwrite or (not os.path.exists(out)):
                self.compute_gscore(pv, out)

        if all_energy:
            print('Extracting all glide related scores.')
            for pv in pvs:
                out = self.path('all_gscores', pv=pv)
                if overwrite or (not os.path.exists(out)):
                    self.compute_all_glide_related_score(pv, out)

        print('Extracting names.')
        for pv in pvs:
            out = self.path('name', pv=pv)
            if overwrite or (not os.path.exists(out)):
                self.compute_name(pv, out)

        if ifp:
            print('Computing interaction fingerprints.')
            unfinished = []
            for pv in pvs:
                out = self.path('ifp', pv=pv)
                if overwrite or (not os.path.exists(out)):
                    unfinished += [(pv, out)]
            run(self.compute_ifp, unfinished, processes)

    def compute_pair_features(self, pvs, processes=1, ifp=True, shape=True, mcss=True, run=mp):
        if len(pvs) == 1:
            return

        mkdir(self.root)

        # Maintain convention that pvs is a list of lists. This is helpful when
        # computing features for a set of potentially overlapping sets of
        # ligands. Only pairwise terms between docking results in the same
        # sub-list will be computed.
        if type(pvs[0]) == str:
            pvs = [pvs]

        pvs = [[os.path.abspath(pv) for pv in _pvs] for _pvs in pvs]

        # Applies the given fxn to all pairs of docking results.
        # Don't include None return values. These correspond to completed jobs.
        def map_pairs(fxn):
            x = set()
            for _pvs in pvs:
                for i, pv1 in enumerate(_pvs):
                    for pv2 in _pvs[i+1:]:
                        _x = fxn(pv1, pv2)
                        if _x is not None:
                            x.add(_x)
            return x

        if ifp:
            print('Computing interaction similarities.')
            mkdir(self.path('ifp-pair', base=True))
            for feature in self.ifp_features:
                def f(pv1, pv2):
                    ifp1 = self.path('ifp', pv=pv1)
                    ifp2 = self.path('ifp', pv=pv2)

                    out = self.path(feature, pv=pv1, pv2=pv2)
                    if not os.path.exists(out):
                        return (ifp1, ifp2, feature, out)
                unfinished = map_pairs(f)
                run(self.compute_ifp_pair, unfinished, processes)

        if shape:
            print('Computing shape similarities.')
            mkdir(self.path('shape', base=True))
            def f(pv1, pv2):
                out = self.path('shape', pv=pv1, pv2=pv2)
                if not os.path.exists(out):
                    return (pv1, pv2, out)
            unfinished = map_pairs(f)
            run(self.compute_shape, unfinished, processes)

        if mcss:
            print('Computing mcss similarities.')
            mkdir(self.path('mcss', base=True))
            def f(pv1, pv2):
                out = self.path('mcss', pv=pv1, pv2=pv2)
                if not os.path.exists(out):
                    return (pv1, pv2, out)
            unfinished = map_pairs(f)
            run(self.compute_mcss, unfinished, processes)

    def compute_name(self, pv, out):
        names = []
        with StructureReader(pv) as sts:
            next(sts)
            for st in sts:
                names += [st.property['s_m_title']]
                if len(names) == self.max_poses:
                    break
        if len(names) > 0:
            np.save(out, names)

    def compute_gscore(self, pv, out):
        gscores = []
        with StructureReader(pv) as sts:
            next(sts)
            for st in sts:
                gscores += [st.property['r_i_docking_score']]
                if len(gscores) == self.max_poses:
                    break
        if len(gscores) > 0:
            np.save(out, gscores)

    def compute_all_glide_related_score(self, pv, out):
        scores = []
        with StructureReader(pv) as sts:
            next(sts)
            for st in sts:
                data = {}
                data['r_i_docking_score'] = st.property['r_i_docking_score']
                for k, v in st.property.items():
                    if ('_glide' in k) or ('r_epik_' in k):
                        data[k] = v
                scores.append(data)
                if len(scores) == self.max_poses:
                    break
        if len(scores) > 0:
            np.save(out, scores)

    def compute_ifp(self, pv, out):
        from features.ifp import ifp
        settings = IFP[self.ifp_version]
        return ifp(settings, pv, out, self.max_poses)

    def compute_ifp_pair(self, ifp1, ifp2, feature, out):
        from features.ifp_similarity import ifp_tanimoto
        tanimotos = ifp_tanimoto(ifp1, ifp2, feature)
        np.save(out, tanimotos)

    def compute_shape(self, pv1, pv2, out):
        from features.shape import shape
        sims = shape(pv2, pv1, version=self.shape_version, max_poses=self.max_poses).T
        np.save(out, sims)

    def compute_mcss(self, pv1, pv2, out):
        from features.mcss import mcss
        rmsds = mcss(pv1, pv2, self.mcss_file, self.max_poses)
        np.save(out, rmsds)
