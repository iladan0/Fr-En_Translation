import numpy as np
import os
import pandas as pd



root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')


def generate_dataset(samples,test_size = 0.2):
    nums, counts = np.unique(samples[:, 0], return_counts=True)
    samples = samples[counts[np.searchsorted(nums, samples[:, 0])] > 1]

    '''
        An Equivalent way to do this is :

            bins = np.bincount(samples[:, 0])
            print(bins[samples[:, 0]])
            samples = samples[bins[samples[:, 0]] > 1]


        ----------------------------------------------------------
        This Code is doing the same work as the two previous Lines
        but in a trivial way (we count each element in axis 0)
        and then we iterate throw all elements and remove those
        having count 1
        ----------------------------------------------------------

        mx = [x[0] for x in list(map(tuple,samples))]
        cpt = collections.Counter(mx)
        cpt = [ u for (u,c) in cpt.items() if c == 1]
        for x in cpt:
            samples = samples[np.where(samples[:,0] != x)]
    '''
    nb_test_samples = int(np.ceil(samples.shape[0]*test_size))
    np.random.shuffle(samples)

    X_test = samples[:nb_test_samples]
    X_train = samples[nb_test_samples:]

    return X_train,X_test



def load_dataset():
    print('loading..')
    df_train = pd.read_csv(os.path.join(root_path,'X_train.csv'))
    df_test  = pd.read_csv(os.path.join(root_path,'X_test.csv'))
    print('loading done.')
    X_train , X_test = df_train.to_numpy(),df_test.to_numpy()
    return X_train,X_test

def save_dataset(filename='ratings.csv'):

    print('loading..')
    df = pd.read_csv(os.path.join(root_path,filename),usecols=['userId','movieId','rating'])
    print('done.')

    samples = df.to_numpy()
    X_train , X_test = generate_dataset(samples,test_size=0.05)
    X_train = pd.DataFrame(X_train,columns=['userId','movieId','rating'])
    X_test =pd.DataFrame(X_test,columns=['userId','movieId','rating'])


    X_train = X_train.astype({'userId': 'int32','movieId': 'int32'})
    X_test = X_test.astype({'userId': 'int32','movieId': 'int32'})

    X_train.to_csv(os.path.join(root_path,"X_train.csv"),index=False)
    X_test.to_csv(os.path.join(root_path,"X_test.csv"),index=False)
    return





if __name__ == '__main__':
    load_dataset()



