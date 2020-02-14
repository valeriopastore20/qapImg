import gym
import numpy as np
import fis_generator as fisg
import random
import os

from gym import spaces

class QapImgEnv(gym.Env):

    #NUMPROD = 10
    #NUMLOC = 120
    #MAXDIST = 8
    #MAXFQ = 1000
   # HIGH = NUMPROD*MAXDIST*MAXFQ
   # MAXMOVES = NUMPROD+10


    def __init__(self):
        #genera e legge i frequent item sets
        path = os.getenv("HOME")+"/fisFolder/fisFile.txt"
        self.matrix_fq = fisg.readFisFile(path)
        self.matrix_fq = self.matrix_fq/np.max(self.matrix_fq)
        self.num_prod = len(self.matrix_fq)
        self.num_loc = self.num_prod
        #inizializza il dizionario delle azioni (In questo modo possiamo avere un action space discreto)
        self.dict = {}
        k=0
        for a in range(self.num_prod):
            for b in range(a,self.num_prod):
                self.dict.update({k : [a,b]})
                k+=1

        # contatore delle mosse effettuate
        self.count = 0
        # Inizializza la matrice dei prodotti
        self.matrix_pl = np.zeros((self.num_prod, self.num_loc), int)
        np.fill_diagonal(self.matrix_pl,1)
        np.random.shuffle(np.transpose(self.matrix_pl))
        #inizializza matrice delle distanze tra locazioni (e' quadrata simmetrica e sulla diagonale c'e' la distanza con l'uscita)
        self.matrix_dist = np.zeros((self.num_loc, self.num_loc), int)
        for i in range(0,self.num_loc):
            for j in range(i+1,self.num_loc):
                self.matrix_dist[i,j] = self.matrix_dist[j,i] = j-i
        for i in range(self.num_loc):
            self.matrix_dist[i,i] = i
        self.matrix_dist = self.matrix_dist/np.max(self.matrix_dist)
        #Crea la matrice finale (l'osservazione su cui opera l'agente)
        matrix_dp = np.dot(np.dot(self.matrix_pl,self.matrix_dist),np.transpose(self.matrix_pl))
        self.matrix_wd = matrix_dp*self.matrix_fq
        self.matrix_wd *= (255/self.matrix_wd.max())
        self.matrix_wd = self.matrix_wd.astype(int)

        self.current_sum = np.sum(self.matrix_wd)
        self.initial_sum = np.sum(self.matrix_wd)

        self.action_space = spaces.Discrete(len(self.dict))
        self.observation_space = spaces.Box(low=0, high=255,shape=(self.num_prod, self.num_prod, 1), dtype=np.uint8)


        self.mff_sum = self.compute_mff_sum(matrix_dp)

    def reset(self):
        self.__init__()
        return np.reshape(self.matrix_wd,(self.num_prod,self.num_prod,1))


    def render(self):
        #np.set_printoptions(threshold=3000)
        print("R E N D E R")
        #print(self.matrix_wd)
        print("INITIAL SUM: {0:.2f}".format(self.initial_sum))
        print("CURRENT SUM: {0:.2f}".format(self.current_sum))
        print("CURRENT IMPROVEMENT: {0:.2f}%".format((self.initial_sum-self.current_sum)/self.initial_sum*100))
        print("MFF SUM: {0:.2f}".format(self.mff_sum))
        print("MFF IMPROVEMENT: {0:.2f}%".format((self.initial_sum-self.mff_sum)/self.initial_sum*100))
        print("R E N D E R")

    def step(self,actionKey):
        done = False
        #converte il valore dell'action nella corrispondente azione
        action = self.dict[actionKey]
        # effettua lo swap sulla matrice di prodotto e ricalcola la matrice finale
        self.matrix_pl[[action[0], action[1]]] = self.matrix_pl[[action[1], action[0]]]
        matrix_dp = np.dot(np.dot(self.matrix_pl,self.matrix_dist),np.transpose(self.matrix_pl))
        self.matrix_wd = matrix_dp*self.matrix_fq
        self.matrix_wd *= (255/self.matrix_wd.max())
        self.matrix_wd = self.matrix_wd.astype(int)
        sum = np.sum(self.matrix_wd)
        #calcola il reward come differenza tra la somma precedente e la somma ottenuta ora. Quindi se questo valore e' positivo vuol dire che la somma totale
        # e' stata ridotta, altrimenti e' stata aumentata
        reward = self.current_sum - sum
        self.current_sum = sum
        self.count+=1
        if(self.count > self.num_prod+10):
            done = True
        return np.reshape(self.matrix_wd,(self.num_prod,self.num_prod,1)), reward, done, {}


# UTILITY METHODS

    def compute_mff_sum(self,matrix):
        diag = np.diag(matrix)
        diag.setflags(write=1)
        min_ind = np.argmin(diag,0)
        matrix_mff = self.matrix_pl[min_ind]
        diag[min_ind] = 90
        for i in range(1,self.num_prod):
            min_ind = np.argmin(diag,0)
            if diag[min_ind] == 90:
                break
            matrix_mff = np.vstack((matrix_mff,self.matrix_pl[min_ind]))
            diag[min_ind] = 90
        matrix_dp = np.dot(np.dot(matrix_mff,self.matrix_dist),np.transpose(matrix_mff))
        matrix_wd = matrix_dp*self.matrix_fq
        mff_sum = np.sum(matrix_wd)
        return mff_sum