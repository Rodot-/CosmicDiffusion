from urllib import request
import re
import h5py
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

BASE_URL = "https://portal.nersc.gov/project/m3363/cosmoUniverse_2019_05_4parE/"
SCRATCH = "./test_images"

def iter_hdf():

    with request.urlopen(BASE_URL) as url:
        body = url.read()
        links = re.findall('href="(\d+)/"', body.decode('utf-8'))
        print('subdirs:', links)
    for link in links:
        with request.urlopen('/'.join((BASE_URL, link))) as url:
            body = url.read()
            sub_links = re.findall('href="(.*\.hdf5)"', body.decode('utf-8'))
            print('sublinks:', len(sub_links))
            for sub_link in sub_links:
                new_link = '/'.join((BASE_URL, link, sub_link))
                with request.urlopen(new_link) as file_url:
                    fname = '/'.join((SCRATCH, link + '_' + sub_link))
                    with open(fname, 'wb') as f:
                        f.write(file_url.read())
                    yield None#h5py.File(fname, 'r', driver='core')

def hdf_download(N=500):

    n = 0
    with ThreadPoolExecutor(16) as tpe:
        futures = []
        with request.urlopen(BASE_URL) as url:
            body = url.read()
            links = re.findall('href="(\d+)/"', body.decode('utf-8'))
            print('subdirs:', links)
        for link in links:
            with request.urlopen('/'.join((BASE_URL, link))) as url:
                body = url.read()
                sub_links = re.findall('href="(.*\.hdf5)"', body.decode('utf-8'))
                print('sublinks:', len(sub_links))
                for sub_link in sub_links:
                    future = tpe.submit(download_link, link, sub_link)
                    futures.append(future)
                    n += 1
                    if n >= N:
                        break
                if n >= N:
                    break

        for future in as_completed(futures):
            print(future.result())
                             
def download_link(link, sub_link):
    new_link = '/'.join((BASE_URL, link, sub_link))
    fname = '/'.join((SCRATCH, link + '_' + sub_link))
    if os.path.exists(fname):
        return fname + ' Already Exists'
    with request.urlopen(new_link) as file_url:
        with open(fname, 'wb') as f:
                f.write(file_url.read())
    return fname

def make_some_data(N=1000, M=50):

    for i, dataset in zip(range(N), iter_hdf()):
        print(i, '/', N)
        continue
        full_frame = dataset['full'][::]
        for j in range(M):
            a = np.random.randint(0, 512-128)
            b = np.random.randint(0, 512-128)
            c = np.random.randint(512)
            subslice = full_frame[a:a+128, b:b+128, c, :].astype('uint8')
            fname = '/'.join((SCRATCH, '_'.join(('frame', str(i), str(j)))+'.png'))
            print(fname)
            #print(subslice.shape, subslice.max(), subslice.min())
            Image.fromarray(subslice, 'RGBA').save(fname)





if __name__ == '__main__':

    hdf_download(1000)

    #make_some_data()

    #my_file = next(iter_hdf())
    #print(my_file)
    #print(my_file.keys())
    #for key in my_file.keys():
    #    print(key, ':', my_file[key].keys() if hasattr(my_file[key], 'keys') else my_file[key])
    #     print(key, ':', my_file[key].keys() if hasattr(my_file[key], 'keys') else my_file[key][::])
