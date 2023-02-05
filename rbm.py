import scipy as sp
import numpy as np

class RBM:
    def __init__(self,p,q, std_w=0.01):
        """Initialize weights and bias of RBM

        Parameters
            Dimension of visible variables
                p: int 
            Dimension of hidden variables
                q: int
            Standard deviation of weights
                std_w: float
        """
        self.a=np.zeros(p)
        self.b=np.zeros(q)
        self.w=np.random.normal(scale=std_w, size=(p,q))
        self.p=p
        self.q=q

    def entree_sortie(self, X):
        """Encode visible variables into hidden variables

        Parameters
            Data array of n visible variables
                X: np.array(n,p)
                
        Returns
            Data array of n hidden variables
                H: np.array(n,q)
        """
        H=1/(1+np.exp(-(X@self.w+self.b)))
        return H

    def sortie_entree(self,H):
        """Decode hidden variables into visible variables

        Parameters
            Data array of n hidden variables
                H: np.array(n,q)

        Returns
            Data array of n visible variables
                X: np.array(n,p)
        """
        X=1/(1+np.exp(-(H @ self.w.T +self.a)))
        return X

    def train_RBM(self, X, eps, nb_epoch, taille_batch):
        """Train Restricted Boltzmann Machine

        Parameters
            Input data
                X: np.array 
            Learning rate
                eps: float
            Number of epochs
                nb_epoch: int
            Size of batch for gradient ascent step
                taille_batch: int
        """
        for epoch in range(nb_epoch):
            #Random permutation of input data
            X=np.random.permutation(X)

            for batch in range(0,np.size(X,0),taille_batch):
                #Create Batch
                X_batch = X[batch: min(batch+taille_batch, np.size(X,0))]
                #Taille batch
                tb=np.size(X_batch,0)
                
                #Create v_0
                v_0=X_batch

                #Encode into hidden space and sample hidden variable
                p_h_v_0 = self.entree_sortie(v_0)
                h_0=(np.random.rand(tb,self.q)<p_h_v_0)*1

                #Decode into visible space and sample visible variable
                p_v_h_0 = self.sortie_entree(h_0)
                v_1=(np.random.rand(tb,self.p)<p_v_h_0)*1

                p_h_v_1 = self.entree_sortie(v_1)

                #Calculate gradients
                grad_w = v_0.T@p_h_v_0 - v_1.T@p_h_v_1
                grad_a = np.sum(v_0 - v_1,axis=0)
                grad_b = np.sum(p_h_v_0 - p_h_v_1,axis=0)

                #Gradient ascend
                self.w += eps/tb*grad_w
                self.a += eps/tb*grad_a
                self.b += eps/tb*grad_b
            
    def generer_image_RBM(self,nb_data,nb_iter_gibbs):
        """Generate image using Gibbs sampler
        
        Parameters
            Number of data samples to generate
                nb_data: int
            Number of iterations of Gibbs sampler
                nb_iter_gibbs: int

        Returns
            Array of Images
                images: np.array(nb_data,20,16)
        """
        v=(np.random.rand(nb_data,self.p)<1/2)*1
        for iter in range(nb_iter_gibbs):
            h=(np.random.rand(nb_data,self.q)<self.entree_sortie(v))*1
            v=(np.random.rand(nb_data,self.p)<self.sortie_entree(h))*1

        images=np.reshape(v,(nb_data,20,16))
        return images