import os
import bnpy
import numpy as np
import torch
from itertools import cycle
from bnpy.data.XData import XData
from matplotlib import pylab




class BNPModel:
    def __init__(self, 
                 save_dir, 
                 gamma0=5.0, 
                 num_lap=20,
                 sF=0.1,
                 birth_kwargs=None, 
                 merge_kwargs=None):
        super(BNPModel, self).__init__()
        # for DPMM model
        self.model = None
        self.info_dict = None
        self.iterator = cycle(range(2))

        self.save_dir = save_dir
        self.num_lap = num_lap
        self.sF = sF
        if not os.path.exists(os.path.join(self.save_dir, 'birth_debug')):
            os.makedirs(os.path.join(self.save_dir, 'birth_debug'))
        if not os.path.exists(os.path.join(self.save_dir, 'data')):
            os.makedirs(os.path.join(self.save_dir, 'data'))

        self.birth_kwargs = birth_kwargs
        self.merge_kwargs = merge_kwargs
        self.gamma0 = gamma0  # concentration parameter of  the DP process
        # self.birth_kwargs = dict(
        #     b_startLap=1,
        #     b_stopLap=2,
        #     b_Kfresh=2,
        #     b_minNumAtomsForNewComp=16.0,
        #     b_minNumAtomsForTargetComp=16.0,
        #     b_minNumAtomsForRetainComp=16.0,
        #     b_minPercChangeInNumAtomsToReactivate=0.1,
        #     b_debugOutputDir=None,   #os.path.join(self.save_dir, 'birth_debug'),  # for debug
        #     b_debugWriteHTML=0,  # for debug
        # )
        #
        # self.merge_kwargs = dict(
        #     m_startLap=2,
        #     # Set limits to number of merges attempted each lap.
        #     # This value specifies max number of tries for each cluster
        #     # Setting this very high (to 50) effectively means try all pairs
        #     m_maxNumPairsContainingComp=50,
        #     # Set "reactivation" limits
        #     # So that each cluster is eligible again after 10 passes thru dataset
        #     # Or when it's size changes by 400%
        #     m_nLapToReactivate=1,
        #     # Specify how to rank pairs (determines order in which merges are tried)
        #     # 'obsmodel_elbo' means rank pairs by improvement to observation model ELBO
        #     m_pair_ranking_procedure='obsmodel_elbo',
        #     # 'total_size' and 'descending' means try largest combined clusters first
        #     # m_pair_ranking_procedure='total_size',
        #     m_pair_ranking_direction='descending',
        # )

    def fit(self, z):
        '''
        fit the model, input z should be torch.tensor format
        '''
        z = XData(z.detach().cpu().numpy())
        if not self.model:
            print("=== Initialing DPMM model ===")
            self.model, self.info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB',
                                                  output_path=os.path.join(self.save_dir,
                                                                           str(next(self.iterator))),
                                                  initname='randexamples',
                                                  K=1, gamma0=self.gamma0,
                                                  sF=self.sF, ECovMat='eye',
                                                  moves='birth,merge', nBatch=5, nLap=self.num_lap,
                                                  **dict(
                                                      sum(map(list, [self.birth_kwargs.items(),
                                                                     self.merge_kwargs.items()]), []))
                                                  )
            print("=== DPMM model initialized ===")
        else:
            self.model, self.info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB',
                                                  output_path=os.path.join(self.save_dir,
                                                                           str(next(self.iterator))),
                                                  initname=self.info_dict['task_output_path'],
                                                  K=self.info_dict['K_history'][-1], gamma0=self.gamma0,
                                                  sF=self.sF, ECovMat='eye',
                                                  moves='birth,merge', nBatch=5, nLap=self.num_lap,
                                                  **dict(
                                                      sum(map(list, [self.birth_kwargs.items(),
                                                                     self.merge_kwargs.items()]), []))
                                                  )
        self.calc_cluster_component_params()


    def plot_clusters(self, z, suffix=""):
        # save best model for debugging
        cur_model, lap_val = bnpy.load_model_at_lap(self.info_dict['task_output_path'], None)
        bnpy.viz.PlotComps.plotCompsFromHModel(cur_model, Data=z)
        pylab.savefig(os.path.join(self.save_dir+'/birth_debug', "dpmm_" + suffix + ".png"))

    def calc_cluster_component_params(self):
        self.comp_mu = [torch.Tensor(self.model.obsModel.get_mean_for_comp(i))
                        for i in np.arange(0, self.model.obsModel.K)]
        self.comp_var = [torch.Tensor(np.sum(self.model.obsModel.get_covar_mat_for_comp(i), axis=0))
                         for i in np.arange(0, self.model.obsModel.K)]

    def kl_divergence_diagonal_gaussian(self, mu_1, mu_2, var_1, var_2):
        """
        var_1: sigma_1 squared
        var_2: sigma_2 squared
        """
        cov_1 = torch.diag_embed(var_1)
        dist_1 = torch.distributions.MultivariateNormal(loc=mu_1, covariance_matrix=cov_1)
        cov_2 = torch.diag_embed(var_2)
        dist_2 = torch.distributions.MultivariateNormal(loc=mu_2, covariance_matrix=cov_2)

        return torch.distributions.kl_divergence(dist_1, dist_2)

    def cluster_assignments(self, z):
        z = XData(z.detach().cpu().numpy())
        LP = self.model.calc_local_params(z)
        # Here, resp is a 2D array of size N x K.
        # Each entry resp[n, k] gives the probability
        # that data atom n is assigned to cluster k under
        # the posterior.
        resp = LP['resp']
        # To convert to hard assignments
        # Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, … K-1, K}.
        Z = resp.argmax(axis=1)
        return resp, Z

    def sample_component(self, num_samples: int, component: int):
        """
        Samples from a dpmm cluster and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        mu = self.comp_mu[component]
        cov = torch.diag_embed(self.comp_var[component])
        dist = torch.distributions.MultivariateNormal(loc=mu,
                                                      covariance_matrix=cov)
        z = dist.sample_n(num_samples)
        return z

    def sample_all(self, num_samples: int):
        """
        Sample a total of (roughly) num_samples samples from all cluster components
        """
        # E_proba_k=model.allocmodel.get_active_comp_probs()
        num_comps = len(self.comp_mu)     # number of active components
        latent_dim = len(self.comp_mu[0])     # dimension of the latent variable
        num_per_comp = int(num_samples/num_comps)
        z = torch.zeros(num_comps * num_per_comp, latent_dim)
        for k in range(0, num_comps):
            z[k * num_per_comp:(k + 1) * num_per_comp, :] = \
                self.sample_component(num_per_comp, k)
        return z


    #################################################
    ############## For VAE evaluation ###############
    #################################################
    def manage_latent_representation(self, z, env_idx, env_name_list:list, prefix:str='', save_info_dict:bool=False):
        '''
        calculate the latent z and their corresponding clusters, save the values for later evaluation
        saved value:
            z: latent encoding from VAE Encoder
            env_idx: corresponding env idx of each z (true label)
            env_name: corresponding env name of each z
            cluster_label: corresponding cluster No. that z should be assigned to
            cluster_param: corresponding cluster parameters (mu & var of diagGauss) of each z
        '''
        
        comps_mu_list = []
        comps_var_list = []
        # cluster label
        _, cluster_label = self.cluster_assignments(z)
        # get clusters param
        comp_mu = [self.model.obsModel.get_mean_for_comp(i) for i in np.arange(0, self.model.obsModel.K)]
        comp_var = [np.sum(self.model.obsModel.get_covar_mat_for_comp(i), axis=0) for i in np.arange(0, self.model.obsModel.K)]
        for i in cluster_label:
            comps_mu_list.append(comp_mu[i])
            comps_var_list.append(comp_var[i])
        # summarize data into dict
        data = dict(
            z=z.detach().cpu().numpy(),
            env_idx=env_idx.detach().cpu().numpy(),
            env_name=env_name_list,
            cluster_label=cluster_label,
            cluster_mu=comps_mu_list,
            cluster_var=comps_var_list,
        )
        # save the file
        np.savez(self.save_dir+'/data'+'/latent_samples_{}.npz'.format(prefix), **data)
        if save_info_dict and self.info_dict is not None:
            self.save_info_dict()


    def load_model(self, abs_path, model_type:str='Best'):
        if os.path.exists(abs_path) and os.path.exists(abs_path+'/data/info_dict.npy'):

            self.info_dict = np.load(abs_path+'/data/info_dict.npy', allow_pickle=True)
            z_files = os.listdir(abs_path+'/data/')
            z_files = [x for x in z_files if x.startswith('latent_samples')]
            if z_files == []:
                print('no latent samples exists at {}'.format(abs_path+'/data/'))
                return

            z_files_split =  [x.split('_')[2] for x in z_files]
            max_index = -1
            max_value = -1
            for i, element in enumerate(z_files_split):
                if element.startswith('step'):
                    _, number = element.split('step')
                    number = int(number)
                    if number > max_value:
                        max_index = i
                        max_value = number
            z_file_name = z_files[max_index]
            z = np.load(abs_path+'/data/'+z_file_name)['z']
            z = XData(z)
            self.model, self.info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB',
                                                  output_path=os.path.join(self.save_dir,
                                                                           str(next(self.iterator))),
                                                  initname=self.info_dict['task_output_path'],
                                                  K=self.info_dict['K_history'][-1], gamma0=self.gamma0,
                                                  sF=self.sF, ECovMat='eye',
                                                  moves='birth,merge', nBatch=5, nLap=self.num_lap,
                                                  **dict(
                                                      sum(map(list, [self.birth_kwargs.items(),
                                                                     self.merge_kwargs.items()]), []))
                                                  )
            self.calc_cluster_component_params()
            print('load {} bnp_model from {}'.format(model_type, abs_path))
        else:
            print(f'invalid bnp_modle load path {abs_path}')

    
    
    def save_info_dict(self):
        info = dict(
            task_output_path = self.info_dict['task_output_path'],
            K_history=[self.info_dict['K_history'][-1]],
        )
        np.save(self.save_dir+'/data'+'/info_dict.npy', info)

    def save_comps_parameters(self):
        '''
        save the model param in form of *npy
        '''
        self.comp_mu = [self.model.obsModel.get_mean_for_comp(i)
                        for i in np.arange(0, self.model.obsModel.K)]
        self.comp_var = [np.sum(self.model.obsModel.get_covar_mat_for_comp(i), axis=0) # save diag value in a 1-dim array
                         for i in np.arange(0, self.model.obsModel.K)]
        data = dict(
            comp_mu=self.comp_mu,
            comp_var=self.comp_var
        )
        np.save(self.save_dir+'/data'+'/comp_params.npy', data)
