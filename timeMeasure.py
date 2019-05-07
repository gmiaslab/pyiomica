import time

def getStartTime():

    return time.time()
    
def getElapsedTime(_start, ref = 0, print_time = True, digits=1):

    if print_time:
        print('Elapsed time: ' + str(round(time.time() - _start, digits)) + ' sec' +'\n')

    return time.time() - _start + ref
