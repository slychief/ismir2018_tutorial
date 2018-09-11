'''

RP_extract: Rhythm Patterns Audio Feature Extractor

@author: 2014-2015 Alexander Schindler, Thomas Lidy


Re-implementation by Alexander Schindler of RP_extract for Matlab
Matlab version originally by Thomas Lidy, based on Musik Analysis Toolbox by Elias Pampalk
( see http://ifs.tuwien.ac.at/mir/downloads.html )

Main function is rp_extract. See function definition and description for more information,
or example usage in main function.

Note: All required functions are provided by the two main scientific libraries numpy and scipy.

Note: In case you alter the code to use transform2mel, librosa needs to be installed: pip install librosa
'''

from __future__ import print_function

import numpy as np

from scipy import stats
from scipy.fftpack import fft
#from scipy.fftpack import rfft #  	Discrete Fourier transform of a real sequence.
from scipy import interpolate

# suppress numpy warnings (divide by 0 etc.)
np.set_printoptions(suppress=True)

# required for debugging
np.set_printoptions(precision=8,
                    threshold=10,
                    suppress=True,
                    linewidth=200,
                    edgeitems=10)


# INITIALIZATION: Constants & Mappings

# Bark Scale
bark = [100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
n_bark_bands = len(bark)

# copy the bark vector (using [:]) and add a 0 in front (to make calculations below easier)
barks = bark[:]
barks.insert(0,0)

# Phone Scale
phon = [3, 20, 40, 60, 80, 100, 101]

# copy the bark vector (using [:]) and add a 0 in front (to make calculations below easier)
phons     = phon[:]
phons.insert(0,0)
phons     = np.asarray(phons)


# Loudness Curves

eq_loudness = np.array([[55,  40, 32, 24, 19, 14, 10,  6,  4,  3,  2,   2, 0,-2,-5,-4, 0,  5, 10, 14, 25, 35], 
                        [66,  52, 43, 37, 32, 27, 23, 21, 20, 20, 20,  20,19,16,13,13,18, 22, 25, 30, 40, 50], 
                        [76,  64, 57, 51, 47, 43, 41, 41, 40, 40, 40,39.5,38,35,33,33,35, 41, 46, 50, 60, 70], 
                        [89,  79, 74, 70, 66, 63, 61, 60, 60, 60, 60,  59,56,53,52,53,56, 61, 65, 70, 80, 90], 
                        [103, 96, 92, 88, 85, 83, 81, 80, 80, 80, 80,  79,76,72,70,70,75, 79, 83, 87, 95,105], 
                        [118,110,107,105,103,102,101,100,100,100,100,  99,97,94,90,90,95,100,103,105,108,115]])

loudn_freq = np.array([31.62, 50, 70.7, 100, 141.4, 200, 316.2, 500, 707.1, 1000, 1414, 1682, 2000, 2515, 3162, 3976, 5000, 7071, 10000, 11890, 14140, 15500])

# We have the loudness values for the frequencies in loudn_freq
# now we calculate in loudn_bark a matrix of loudness sensation values for the bark bands margins

i = 0
j = 0

loudn_bark = np.zeros((eq_loudness.shape[0], len(bark)))

for bsi in bark:

    while j < len(loudn_freq) and bsi > loudn_freq[j]:
        j += 1
    
    j -= 1
    
    if np.where(loudn_freq == bsi)[0].size != 0: # loudness value for this frequency already exists
        loudn_bark[:,i] = eq_loudness[:,np.where(loudn_freq == bsi)][:,0,0]
    else:
        w1 = 1 / np.abs(loudn_freq[j] - bsi)
        w2 = 1 / np.abs(loudn_freq[j + 1] - bsi)
        loudn_bark[:,i] = (eq_loudness[:,j]*w1 + eq_loudness[:,j+1]*w2) / (w1 + w2)
    
    i += 1



# SPECTRAL MASKING Spreading Function
# CONST_spread contains matrix of spectral frequency masking factors

CONST_spread = np.zeros((n_bark_bands,n_bark_bands))

for i in range(n_bark_bands):
    CONST_spread[i,:] = 10**((15.81+7.5*((i-np.arange(n_bark_bands))+0.474)-17.5*(1+((i-np.arange(n_bark_bands))+0.474)**2)**0.5)/10)



# UTILITY FUNCTIONS


def nextpow2(num):
    '''NextPow2

    find the next highest number to the power of 2 to a given number
    and return the exponent to 2
    (analogously to Matlab's nextpow2() function)
    '''

    n = 2
    i = 1
    while n < num:
        n *= 2 
        i += 1
    return i



# FFT FUNCTIONS

def periodogram(x,win,Fs=None,nfft=1024):
    ''' Periodogram

    Periodogram power spectral density estimate
    Note: this function was written with 1:1 Matlab compatibility in mind.

    The number of points, nfft, in the discrete Fourier transform (DFT) is the maximum of 256 or the next power of two greater than the signal length.

    :param x: time series data (e.g. audio signal), ideally length matches nfft
    :param win: window function to be applied (e.g. Hanning window). in this case win expects already data points of the window to be provided.
    :param Fs: sampling frequency (unused)
    :param nfft: number of bins for FFT (ideally matches length of x)
    :return: Periodogram power spectrum (np.array)
    '''


    #if Fs == None:
    #    Fs = 2 * np.pi         # commented out because unused
   
    U  = np.dot(win.conj().transpose(), win) # compensates for the power of the window.
    Xx = fft((x * win),nfft) # verified
    P  = Xx*np.conjugate(Xx)/U
    
    # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
    # Also, compute the corresponding freq vector & freq units.
    
    # Generate the one-sided spectrum [Power] if so wanted
    if nfft % 2 != 0:
        select = np.arange((nfft+1)/2)  # ODD
        P = P[select,:] # Take only [0,pi] or [0,pi)
        P[1:-1] = P[1:-1] * 2 # Only DC is a unique point and doesn't get doubled
    else:
        #select = np.arange(nfft/2+1);    # EVEN
        #P = P[select,:]         # Take only [0,pi] or [0,pi) # TODO: why commented out?
        P[1:-2] = P[1:-2] * 2

    P = P / (2 * np.pi)

    return P




def calc_spectrogram(wavsegment,fft_window_size,fft_overlap = 0.5,real_values=True):

    ''' Calc_Spectrogram

    calculate spectrogram using periodogram function (which performs FFT) to convert wave signal data
    from time to frequency domain (applying a Hanning window and (by default) 50 % window overlap)

    :param wavsegment: audio wave file data for a segment to be analyzed (mono (i.e. 1-dimensional vector) only
    :param fft_window_size: windows size to apply FFT to
    :param fft_overlap: overlap to apply during FFT analysis in % fraction (e.g. default = 0.5, means 50% overlap)
    :param real_values: if True, return real values by taking abs(spectrogram), if False return complex values
    :return: spectrogram matrix as numpy array (fft_window_size, n_frames)
    '''

    # hop_size (increment step in samples, determined by fft_window_size and fft_overlap)
    hop_size = int(fft_window_size*(1-fft_overlap))

    # this would compute the segment length, but it's pre-defined above ...
    # segment_size = fft_window_size + (frames-1) * hop_size
    # ... therefore we convert the formula to give the number of frames needed to iterate over the segment:
    n_frames = int((wavsegment.shape[0] - fft_window_size) / hop_size + 1)
    # n_frames_old = wavsegment.shape[0] / fft_window_size * 2 - 1  # number of iterations with 50% overlap

    # TODO: provide this as parameter for better caching?
    han_window = np.hanning(fft_window_size) # verified

    # initialize result matrix for spectrogram
    spectrogram  = np.zeros((fft_window_size, n_frames), dtype=np.complex128)

    # start index for frame-wise iteration
    ix = 0

    for i in range(n_frames): # stepping through the wave segment, building spectrum for each window
        spectrogram[:,i] = periodogram(wavsegment[ix:ix+fft_window_size], win=han_window,nfft=fft_window_size)
        ix = ix + hop_size

        # NOTE: tested scipy periodogram BUT it delivers totally different values AND takes 2x the time of our periodogram function (0.13 sec vs. 0.06 sec)
        # from scipy.signal import periodogram # move on top
        #f,  spec = periodogram(x=wavsegment[idx],fs=samplerate,window='hann',nfft=fft_window_size,scaling='spectrum',return_onesided=True)

    if real_values: spectrogram = np.abs(spectrogram)

    return spectrogram


# FEATURE FUNCTIONS

def calc_statistical_features(matrix):

    result = np.zeros((matrix.shape[0],7))
    
    result[:,0] = np.mean(matrix, axis=1)
    result[:,1] = np.var(matrix, axis=1, dtype=np.float64) # the values for variance differ between MATLAB and Numpy!
    result[:,2] = stats.skew(matrix, axis=1)
    result[:,3] = stats.kurtosis(matrix, axis=1, fisher=False) # Matlab calculates Pearson's Kurtosis
    result[:,4] = np.median(matrix, axis=1)
    result[:,5] = np.min(matrix, axis=1)
    result[:,6] = np.max(matrix, axis=1)
    
    result[np.where(np.isnan(result))] = 0
    
    return result


# PSYCHO-ACOUSTIC TRANSFORMS as individual functions


# Transform 2 Mel Scale: NOT USED by rp_extract, but included for testing purposes or for import into other programs

def transform2mel(spectrogram,samplerate,fft_window_size,n_mel_bands = 80,freq_min = 0,freq_max = None):
    '''Transform to Mel

    convert a spectrogram to a Mel scale spectrogram by grouping original frequency bins
    to Mel frequency bands (using Mel filter from Librosa)

    Parameters
    spectrogram: input spectrogram
    samplerate: samplerate of audio signal
    fft_window_size: number of time window / frequency bins in the FFT analysis
    n_mel_bands: number of desired Mel bands, typically 20, 40, 80 (max. 128 which is default when 'None' is provided)
    freq_min: minimum frequency (Mel filters will be applied >= this frequency, but still return n_meld_bands number of bands)
    freq_max: cut-off frequency (Mel filters will be applied <= this frequency, but still return n_meld_bands number of bands)

    Returns:
    mel_spectrogram: Mel spectrogram: np.array of shape(n_mel_bands,frames) maintaining the number of frames in the original spectrogram
    '''

    from librosa.filters import mel

    # Syntax: librosa.filters.mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False)
    mel_basis = mel(samplerate,fft_window_size, n_mels=n_mel_bands,fmin=freq_min,fmax=freq_max)

    freq_bin_max = mel_basis.shape[1] # will be fft_window_size / 2 + 1

    # IMPLEMENTATION WITH FOR LOOP
    # initialize Mel Spectrogram matrix
    #n_mel_bands = mel_basis.shape[0]  # get the number of bands from result in case 'None' was specified as parameter
    #mel_spectrogram = np.empty((n_mel_bands, frames))

    #for i in range(frames): # stepping through the wave segment, building spectrum for each window
    #    mel_spectrogram[:,i] = np.dot(mel_basis,spectrogram[0:freq_bin_max,i])

    # IMPLEMENTATION WITH DOT PRODUCT (15% faster)
    # multiply the mel filter of each band with the spectogram frame (dot product executes it on all frames)
    # filter will be adapted in a way so that frequencies beyond freq_max will be discarded
    mel_spectrogram = np.dot(mel_basis,spectrogram[0:freq_bin_max,:])
    return mel_spectrogram




# Bark Transform: Convert Spectrogram to Bark Scale
# matrix: Spectrogram values as returned from periodogram function (see calc_spectrogram) - but only relevant part:
# i.e. n_frequencies / 2 + 1 (equals fft_window_size / 2 + 1)
# freq_axis: array of frequency values along the frequency axis
# max_bands: limit number of Bark bands (1...24) (counting from lowest band)
def transform2bark(matrix, freq_axis, max_bands=None):

    # barks and n_bark_bands have been initialized globally above

    if max_bands == None:
        max_band = n_bark_bands
    else:
        max_band = min(n_bark_bands,max_bands)

    matrix_out = np.zeros((max_band,matrix.shape[1]),dtype=matrix.dtype)

    for b in range(max_band):
        # consider (and sum up) those frequencies that lie between the defined bark band limits
        freq_range_bool = (freq_axis >= barks[b]) & (freq_axis < barks[b + 1])
        # debug print
        #print "Analyzing Bark", b, "from", barks[b], "to", barks[b + 1], \
        #    "Hz (frequency bands", freq_axis[freq_range_bool][0], "to", freq_axis[freq_range_bool][-1], "Hz)"
        matrix_out[b] = np.sum(matrix[freq_range_bool], axis=0)

    return matrix_out

# Spectral Masking (assumes values are arranged in <=24 Bark bands)
def do_spectral_masking(matrix):

    n_bands = matrix.shape[0]

    # CONST_spread has been initialized globally above
    spread = CONST_spread[0:n_bands,0:n_bands] # not sure if column limitation is right here; was originally written for n_bark_bands = 24 only
    matrix = np.dot(spread, matrix)
    return(matrix)

# Map to Decibel Scale
def transform2db(matrix):
    '''Map to Decibel Scale'''
    matrix[np.where(matrix < 1)] = 1
    matrix = 10 * np.log10(matrix)
    return matrix

# Transform to Phon (assumes matrix is in dB scale)
def transform2phon(matrix):

    old_npsetting = np.seterr(invalid='ignore') # avoid 'RuntimeWarning: invalid value encountered in divide' at ifac division below

    # number of bark bands, matrix length in time dim
    n_bands = matrix.shape[0]
    t       = matrix.shape[1]

    # DB-TO-PHON BARK-SCALE-LIMIT TABLE
    # introducing 1 level more with level(1) being infinite
    # to avoid (levels - 1) producing errors like division by 0

    #%%table_dim = size(CONST_loudn_bark,2);
    table_dim = n_bands; # OK
    cbv       = np.concatenate((np.tile(np.inf,(table_dim,1)), loudn_bark[:,0:n_bands].transpose()),1) # OK

    # init lowest level = 2
    levels = np.tile(2,(n_bands,t)) # OK

    for lev in range(1,6): # OK
        db_thislev = np.tile(np.asarray([cbv[:,lev]]).transpose(),(1,t))
        levels[np.where(matrix > db_thislev)] = lev + 2

    # the matrix 'levels' stores the correct Phon level for each data point
    cbv_ind_hi = np.ravel_multi_index(dims=(table_dim,7), multi_index=np.array([np.tile(np.array([range(0,table_dim)]).transpose(),(1,t)), levels-1]), order='F')
    cbv_ind_lo = np.ravel_multi_index(dims=(table_dim,7), multi_index=np.array([np.tile(np.array([range(0,table_dim)]).transpose(),(1,t)), levels-2]), order='F')

    # interpolation factor % OPT: pre-calc diff
    ifac = (matrix[:,0:t] - cbv.transpose().ravel()[cbv_ind_lo]) / (cbv.transpose().ravel()[cbv_ind_hi] - cbv.transpose().ravel()[cbv_ind_lo])

    ifac[np.where(levels==2)] = 1 # keeps the upper phon value;
    ifac[np.where(levels==8)] = 1 # keeps the upper phon value;

    # phons has been initialized globally above

    matrix[:,0:t] = phons.transpose().ravel()[levels - 2] + (ifac * (phons.transpose().ravel()[levels - 1] - phons.transpose().ravel()[levels - 2])) # OPT: pre-calc diff

    np.seterr(invalid=old_npsetting['invalid']) # restore RuntimeWarning setting for np division error

    return(matrix)


# Transform to Sone scale (assumes matrix is in Phon scale)
def transform2sone(matrix):
    idx     = np.where(matrix >= 40)
    not_idx = np.where(matrix < 40)

    matrix[idx]     =  2**((matrix[idx]-40)/10)    #
    matrix[not_idx] =  (matrix[not_idx]/40)**2.642 # max => 438.53
    return(matrix)


# MAIN Rhythm Pattern Extraction Function

def rp_extract( wavedata,                          # pcm (wav) signal data normalized to (-1,1)
                samplerate,                    # signal sampling rate

                # which features to extract
                extract_rp   = False,          # extract Rhythm Patterns features
                extract_ssd  = False,          # extract Statistical Spectrum Descriptor
                extract_tssd = False,          # extract temporal Statistical Spectrum Descriptor
                extract_rh   = False,          # extract Rhythm Histogram features
                extract_rh2  = False,          # extract Rhythm Histogram features including Fluctuation Strength Weighting
                extract_trh  = False,          # extract temporal Rhythm Histogram features
                extract_mvd  = False,          # extract modulation variance descriptor

                # processing options
                skip_leadin_fadeout =  1,      # >=0  how many sample windows to skip at the beginning and the end
                step_width          =  1,      # >=1  each step_width'th sample window is analyzed
                n_bark_bands        = 24,      # 2..24 number of desired Bark bands (from low frequencies to high) (e.g. 15 or 20 or 24 for 11, 22 and 44 kHz audio respectively) (1 delivers undefined output)
                mod_ampl_limit      = 60,      # 2..257 number of modulation frequencies on x-axis
                
                # enable/disable parts of feature extraction
                transform_bark                 = True,  # [S2] transform to Bark scale
                spectral_masking               = True,  # [S3] compute Spectral Masking
                transform_db                   = True,  # [S4] transfrom to dB: advisable only to turn off when [S5] and [S6] are turned off too
                transform_phon                 = True,  # [S5] transform to Phon: if disabled, Sone_transform will be disabled too
                transform_sone                 = True,  # [S6] transform to Sone scale (only applies if transform_phon = True)
                fluctuation_strength_weighting = True,  # [R2] apply Fluctuation Strength weighting curve
                #blurring                       = True  # [R3] Gradient+Gauss filter # TODO: not yet implemented

                return_segment_features = False,     # this will return features per each analyzed segment instead of aggregated ones
                verbose = False                      # print messages whats going on
                ):

    '''Rhythm Pattern Feature Extraction

    performs segment-wise audio feature extraction from provided audio wave (PCM) data
    and extracts the following features:

        Rhythm Pattern
        Statistical Spectrum Descriptor
        Statistical Histogram
        temporal Statistical Spectrum Descriptor
        Rhythm Histogram
        temporal Rhythm Histogram features
        Modulation Variance Descriptor

    Examples:
    >>> from audiofile_read import *
    >>> samplerate, samplewidth, wavedata = audiofile_read("music/BoxCat_Games_-_10_-_Epic_Song.mp3") #doctest: +ELLIPSIS
    Decoded .mp3 with: mpg123 -q -w /....wav music/BoxCat_Games_-_10_-_Epic_Song.mp3
    >>> feat = rp_extract(wavedata, samplerate, extract_rp=True, extract_ssd=True, extract_rh=True)
    Analyzing 7 segments
    >>> for k in feat.keys():
    ...     print k.upper() +  ":", feat[k].shape[0], "dimensions"
    SSD: 168 dimensions
    RH: 60 dimensions
    RP: 1440 dimensions
    >>> print feat["rp"]
    [ 0.01599218  0.01979605  0.01564305  0.01674175  0.00959912  0.00931604  0.00937831  0.00709122  0.00929631  0.00754473 ...,  0.02998088  0.03602739  0.03633861  0.03664331  0.02589753  0.02110256
      0.01457744  0.01221825  0.0073788   0.00164668]
    >>> print feat["rh"]
    [  7.11614842  12.58303013   6.96717295   5.24244146   6.49677561   4.21249659  12.43844045   4.19672357   5.30714983   6.1674115  ...,   1.55870044   2.69988854   2.75075831   3.67269877  13.0351257
      11.7871738    3.76106713   2.45225195   2.20457928   2.06494926]
    >>> print feat["ssd"]
    [  3.7783279    5.84444695   5.58439197   4.87849697   4.14983056   4.09638223   4.04971225   3.96152261   3.65551062   3.2857232  ...,  14.45953191  14.6088727   14.03351539  12.84783095  10.81735946
       9.04121124   7.13804008   5.6633501    3.09678286   0.52076428]

    '''


    # PARAMETER INITIALIZATION
    # non-exhibited parameters
    include_DC = False
    FLATTEN_ORDER = 'F' # order how matrices are flattened to vector: 'F' for Matlab/Fortran, 'C' for C order (IMPORTANT TO USE THE SAME WHEN reading+reshaping the features)

    # segment_size should always be ~6 sec, fft_window_size should always be ~ 23ms

    if (samplerate == 11025):
        segment_size    = 2**16
        fft_window_size = 256
    elif (samplerate == 22050):
        segment_size    = 2**17
        fft_window_size = 512
    elif (samplerate == 44100):
        segment_size    = 2**18
        fft_window_size = 1024
    else:
        # throw error not supported
        raise ValueError('A sample rate of ' + str(samplerate) + " is not supported (only 11, 22 and 44 kHz).")
    
    # calculate frequency values on y-axis (for Bark scale calculation):
    # freq_axis = float(samplerate)/fft_window_size * np.arange(0,(fft_window_size/2) + 1)
    # the spectrum result of an FFT is mirrored, so we take only half of its bins + 1 int account:
    n_freq = int(fft_window_size//2) + 1

    # linear space from 0 to samplerate/2 in (fft_window_size/2+1) steps
    freq_axis = np.linspace(0, float(samplerate)/2, n_freq, endpoint=True)

    # CONVERT STEREO TO MONO: Average the channels
    if wavedata.ndim > 1:                    # if we have more than 1 dimension
        if wavedata.shape[1] == 1:           # check if 2nd dimension is just 1
            wavedata = wavedata[:,0]         # then we take first and only channel
        else:
            wavedata = np.mean(wavedata, 1)  # otherwise we average the signals over the channels


    # SEGMENT INITIALIZATION
    # find positions of wave segments
    
    skip_seg = skip_leadin_fadeout
    seg_pos  = np.array([1, segment_size]) # array with 2 entries: start and end position of selected segment

    seg_pos_list = []  # list to store all the individual segment positions (only when return_segment_features == True)

    # if file is too small, don't skip leadin/fadeout and set step_width to 1
    if ((skip_leadin_fadeout > 0) or (step_width > 1)):

        duration =  wavedata.shape[0]/samplerate

        if (duration < 45):
            step_width = 1
            skip_seg   = 0
            # TODO: do this as a warning?
            if verbose: print("Duration < 45 seconds: setting step_width to 1 and skip_leadin_fadeout to 0.")

        else:
            # advance by number of skip_seg segments (i.e. skip lead_in)
            seg_pos = seg_pos + segment_size * skip_seg
    
    # calculate number of segments
    n_segments = int(np.floor( (np.floor( (wavedata.shape[0] - (skip_seg*2*segment_size)) / segment_size ) - 1 ) / step_width ) + 1)
    if verbose: print("Analyzing", n_segments, "segments")

    if n_segments == 0:
        raise ValueError("Not enough data to analyze! Minimum sample length needs to be " +
                         str(segment_size) + " (5.94 seconds) but it is " + str(wavedata.shape[0]) +
                         " (" + str(round(wavedata.shape[0] * 1.0 / samplerate,2)) + " seconds)")

    # initialize output
    features = {}

    ssd_list = []
    rh_list  = []
    rh2_list = []
    rp_list  = []
    mvd_list = []

    hearing_threshold_factor = 0.0875 * (2**15)

    # SEGMENT ITERATION

    for seg_id in range(n_segments):

        # keep track of segment position
        if return_segment_features:
            seg_pos_list.append(seg_pos)
        
        # EXTRACT WAVE SEGMENT that will be processed
        # data is assumed to be mono waveform
        wavsegment = wavedata[seg_pos[0]-1:seg_pos[1]] # verified
        
        # v210715
        # Python : [-0.0269165  -0.02128601 -0.01864624 -0.01893616 -0.02166748 -0.02694702 -0.03457642 -0.04333496 -0.05166626 -0.05891418]
        # Matlab : [-0,0269165  -0,02125549 -0,01861572 -0,01893616 -0,02165222 -0,02694702 -0,03456115 -0,04331970 -0,05166626 -0,05891418]

        
        # adjust hearing threshold # TODO: move after stereo-mono conversion above?
        wavsegment = wavsegment * hearing_threshold_factor

        # v210715
        # Python : [ -77.175    -61.03125  -53.4625   -54.29375  -62.125    -77.2625   -99.1375  -124.25    -148.1375  -168.91875]
        # Matlab : [ -77,175    -60,94375  -53,3750   -54,29375  -62,081    -77,2625   -99,0938  -124,21    -148,1375  -168,91875]        

        matrix = calc_spectrogram(wavsegment,fft_window_size)

        # v210715
        #Python:   0.01372537     0.51454915    72.96077581   84.86663379   2.09940049    3.29631279   97373.2756834      23228.2065494       2678.44451741     30467.235416   
        #      :  84.50635406    58.32826049  1263.82538188  234.11858349  85.48176796   97.26094525  214067.91208223   3570917.53366476   2303291.96676741   1681002.94519665 
        #      : 171.47168402  1498.04129116  3746.45491915  153.01444364  37.20801758  177.74229702  238810.1975412    3064388.50572536   5501187.79635479   4172009.81345923
                                                                         
        #Matlab:   0,01528259     0,49653179    73,32978523   85,38774541   2,00416767    3,36618763   97416,24267209     23239,84650814      2677,01521862     30460,9231041364
        #      :  84,73805309    57,84524803  1263,40594029  235,62185973  85,13826606   97,61122652  214078,02415144   3571346,74831746   2303286,74666381   1680967,41922679
        #      : 170,15377915  1500,98052242  3744,98456435  154,14108817  36,69362260  177,48982263  238812,02171250   3064642,99278220   5501230,26588318   4172058,72803277

        # NOTE that after FFT, the spectrogram contains typically 1024 frequency bands on y-axis
        # but the spectrum is mirrored, so we typically take only (n_fft / 2) + 1 bands (see n_freq above)
        # freq_axis already takes this into account and therefore before calling transform2bark
        # we cut only the relevant half (+1) from the returned spectrogram (before that, we make a double check if the spectrums size)

        if (matrix.shape[0] // 2) + 1 != n_freq:
            raise ValueError(
                "Result shape of returned spectrogram " + str((matrix.shape[0] // 2) + 1) + " does not match " +
                "predefined frequency axis length of " + str(n_freq) + " bins")

        # from the mirrored Spectrum cut only one half + 1 bin
        matrix = matrix[0:n_freq, :]

        # PSYCHO-ACOUSTIC TRANSFORMS

        # Map to Bark Scale
        if transform_bark:
            matrix = transform2bark(matrix,freq_axis,n_bark_bands)

        # v210715
        # Python:    255.991763   1556.884100   5083.2410768    471.9996609   124.789186   278.299555  550251.385306   6658534.245939   7807158.207639  5883479.99407189 
        #       :  77128.354925  10446.109041  22613.8525735  13266.2502432  2593.395039  1367.697057  675114.554043  23401741.536499   6300109.471193  8039710.71759598 
        #       : 127165.795400  91270.354107  15240.3501050  16291.2234730  1413.851495  2166.723800  868138.817452  20682384.237884   8971171.605009  5919089.97818692 

        # Matlab:    254,907114   1559,322302    5081,720289    475,1506933   123,836056    278,46723  550306,288536   6659229,587607   7807194,027765   5883487,07036370
        #       :  77118,196343  10447,961479   22605,559124  13266,4432995  2591,064037   1368,48462  675116,996782  23400723,570438   6300124,132022   8039688,83884099
        #       : 127172,560642  91251,040768   15246,639683  16286,4542687  1414,053166   2166,42874  868063,055613  20681863,052695   8971108,607811   5919136,16752791


        # Spectral Masking
        if spectral_masking:
            matrix = do_spectral_masking(matrix)
            
        # v210715
        # Python:  12978.051641    3416.109125   8769.913963   2648.888265   547.12360   503.50224   660888.17361  10480839.33617   8840234.405272   7193404.23970964 
        #       : 100713.471006   27602.656332  27169.741240  16288.350176  2887.60281  1842.05959  1021358.42618  29229962.41626  10653981.441005  11182818.62910279 
        #       : 426733.607945  262537.326945  43522.106075  41091.381283  4254.39289  4617.45877  1315036.85377  31353824.35688  12417010.121754   9673923.23590653 
        
        # Matlab:  12975,335615     3418,81282   8767,062187   2652,061105   545,79379   503,79683   660943,32199  10481368,76411   8840272,477464   7193407,85259461
        #       : 100704,175421    27602,34142  27161,901160  16288,924458  2884,94883  1842,86020  1021368,99046  29229118,99738  10653999,341989  11182806,7524195
        #       : 426751,992198   262523,89306  43524,970883  41085,415594  4253,42029  4617,35691  1314966,73269  31353021,99155  12416968,806879   9673951,88376021


        # Map to Decibel Scale
        if transform_db:
            matrix = transform2db(matrix)

        # v210715
        # Python: 41.13209498  35.33531736  39.42995333  34.23063639  27.38085455  27.02001413  58.2012798   70.20396064  69.46463781  68.56934467 
        #       : 50.03087564  44.40950878  44.34085502  42.11877097  34.60537456  32.65303677  60.09178176  74.65828257  70.27511936  70.48551281 
        #       : 56.30156848  54.19191059  46.38709903  46.1375074   36.28837595  36.64403027  61.18937924  74.96290521  70.94017035  69.85602637 
        
        # Matlab: 41,13118599  35,33875324  39,42854087  34,23583526  27,37028596  27,02255437  58,20164218  70,20418000  69,46465651  68,56934684
        #       : 50,03047477  44,40945923  44,33960164  42,11892409  34,60138115  32,65492392  60,09182668  74,65815725  70,27512665  70,48550820
        #       : 56,30175557  54,19168835  46,38738489  46,13687684  36,28738298  36,64393446  61,18914765  74,96279407  70,94015590  69,85603922

        
        # Transform Phon
        if transform_phon:
            matrix = transform2phon(matrix)

        # v210715
        # Python: 25.90299283  17.82310731  23.4713619   16.37852452   7.42111749   6.94924924  47.58029453  60.22662293  59.43646085  58.49404702 
        #       : 47.03087564  41.40950878  41.34085502  38.89846372  29.5067182   27.06629597  57.09178176  71.65828257  67.27511936  67.48551281 
        #       : 55.02273887  52.91308099  45.10826943  44.8586778   34.3678058   34.769195    59.91054964  73.68407561  69.66134075  68.57719676 
        
        # Matlab: 25,90169428  17,82760039  23,46934410  16,38532303   7,40729702   6,95257110  47,58067598  60,22686667  59,43648053  58,49404931
        #       : 47,03047477  41,40945923  41,33960164  38,89865511  29,50172644  27,06865491  57,09182668  71,65815725  67,27512665  67,48550820
        #       : 55,02292596  52,91285875  45,10855528  44,85804723  34,36668514  34,76908687  59,91031805  73,68396446  69,66132629  68,57720962


        # Transform Sone
        if transform_sone:
            matrix = transform2sone(matrix)

        # v210715
        # Python: 0.31726931   0.11815598   0.24452297   0.09450863   0.01167179   0.009812     1.6911791    4.06332931   3.84676603   3.60351463 
        #       : 1.62798518   1.10263162   1.09739697   0.92887876   0.44759842   0.35631529   3.26974511   8.97447943   6.62312431   6.72041945 
        #       : 2.83288863   2.44749871   1.42486669   1.40042797   0.669685     0.69054778   3.97527582  10.327417     7.81439442   7.24868691
        
        # Matlab: 0,31722728   0,11823469   0,24446743   0,09461230   0,01161444   0,00982439   1,69122381   4,06339796   3,84677128   3,60351520
        #       : 1,62793994   1,10262783   1,09730163   0,92889083   0,44739839   0,35639734   3,26975529   8,97440147   6,62312765   6,72041730
        #       : 2,83292537   2,44746100   1,42489491   1,40036676   0,66962731   0,69054210   3,97521200  10,32733744   7,81438659   7,24869337


        # FEATURES: now we got a Sonogram and extract statistical features
    
        # SSD: Statistical Spectrum Descriptors
        if extract_ssd or extract_tssd:
            ssd = calc_statistical_features(matrix)
            ssd_list.append(ssd.flatten(FLATTEN_ORDER))

        # v210715
        # Python: 2.97307486   5.10356599   0.65305978   2.35489911   2.439558     0.009812     8.1447095 
        #       : 4.72262845   7.30899976   0.17862996   2.10446264   4.58595337   0.25538117  12.83339251
        #       : 4.77858109   5.52646859   0.23911764   2.9056742    4.96338019   0.589568    13.6683906 
        #       : 4.43503421   3.69422906   0.41473155   3.06743402   4.33220988   0.88354694  10.89393754
        #       : 3.77216546   2.3993334    0.84001713   4.35548197   3.65140589   1.01199696  11.07806891
        #       : 3.60563073   2.09907968   1.49906811   7.07183968   3.35596471   1.00619842  11.2872743 
        #       : 3.56816128   2.20237398   1.69790808   7.57870223   3.33806767   1.10826324  10.84965392
        #       : 3.43734647   2.38648202   1.59655791   6.86704341   3.23361995   1.10198021  11.89470587
        #       : 3.18466303   2.39479532   1.99223131   8.83987184   2.8819031    0.93982524  11.28737448
        #       : 2.90996406   1.85412568   1.97247446   8.36738395   2.68063918   0.81760102   9.64247378
        
        # Matlab: 2,97309758   5,11366933   0,65306558   2,35489605   2,43956735   0,00982439   8,14473582
        #       : 4,72264163   7,32338449   0,17863061   2,10444843   4,58593777   0,25568703  12,83335168
        #       : 4,77859306   5,53731457   0,23911126   2,90567055   4,96338616   0,58959588  13,66839858
        #       : 4,43505068   3,70148292   0,41473410   3,06742263   4,33222037   0,88357883  10,89397920
        #       : 3,77217541   2,40405654   0,84000183   4,35540491   3,65136495   1,01191651  11,07802201
        #       : 3,60563459   2,10319516   1,49905911   7,07181623   3,35609824   1,00628652  11,28728291
        #       : 3,56820841   2,20675908   1,69792784   7,57880557   3,33819690   1,10830805  10,84975850
        #       : 3,43736757   2,39117736   1,59656951   6,86710630   3,23366165   1,10199096  11,89486723
        #       : 3,18467212   2,39951286   1,99223621   8,83991021   2,88200015   0,93978494  11,28733449
        #       : 2,90997546   1,85776617   1,97246361   8,36742039   2,68074853   0,81790606   9,64262886

        # values verified

        # RP: RHYTHM PATTERNS
        feature_part_xaxis1 = range(0,mod_ampl_limit)    # take first (opts.mod_ampl_limit) values of fft result including DC component
        feature_part_xaxis2 = range(1,mod_ampl_limit+1)  # leave DC component and take next (opts.mod_ampl_limit) values of fft result

        if (include_DC):
            feature_part_xaxis_rp = feature_part_xaxis1
        else:
            feature_part_xaxis_rp = feature_part_xaxis2

        # 2nd FFT
        fft_size = 2**(nextpow2(matrix.shape[1]))

        if (mod_ampl_limit >= fft_size):
            raise(ValueError("mod_ampl_limit option must be smaller than FFT window size (" + str(fft_size) +  ")."))
            # NOTE: in fact only half of it (256) makes sense due to the symmetry of the FFT result
        
        rhythm_patterns = np.zeros((matrix.shape[0], fft_size), dtype=np.complex128)
        #rhythm_patterns = np.zeros((matrix.shape[0], fft_size), dtype=np.float64)

        # real_matrix = abs(matrix)

        for b in range(0,matrix.shape[0]):
        
            rhythm_patterns[b,:] = fft(matrix[b,:], fft_size)

            # tried this instead, but ...
            #rhythm_patterns[b,:] = fft(real_matrix[b,:], fft_size)   # ... no performance improvement
            #rhythm_patterns[b,:] = rfft(real_matrix[b,:], fft_size)  # ... different output values
        
        rhythm_patterns = rhythm_patterns / 256  # why 256?

        # convert from complex128 to float64 (real)
        rp = np.abs(rhythm_patterns[:,feature_part_xaxis_rp]) # verified

        # MVD: Modulation Variance Descriptors
        if extract_mvd:
            mvd = calc_statistical_features(rp.transpose()) # verified
            mvd_list.append(mvd.flatten(FLATTEN_ORDER))

        # RH: Rhythm Histograms - OPTION 1: before fluctuation_strength_weighting (as in Matlab)
        if extract_rh or extract_trh:
            rh = np.sum(np.abs(rhythm_patterns[:,feature_part_xaxis2]),axis=0) #without DC component # verified
            rh_list.append(rh.flatten(FLATTEN_ORDER))

        # final steps for RP:

        # Fluctuation Strength weighting curve
        if (extract_rp or extract_rh2) and fluctuation_strength_weighting:

            # modulation frequency x-axis (after 2nd FFT)
            # mod_freq_res = resolution of modulation frequency axis (0.17 Hz)
            mod_freq_res  = 1 / (float(segment_size) / samplerate)

            #  modulation frequencies along x-axis from index 0 to 256)
            mod_freq_axis = mod_freq_res * np.array(feature_part_xaxis_rp)

            #  fluctuation strength curve
            fluct_curve = 1 / (mod_freq_axis/4 + 4/mod_freq_axis)

            for b in range(rp.shape[0]):
                rp[b,:] = rp[b,:] * fluct_curve #[feature_part_xaxis_rp]

        #values verified


        # RH: Rhythm Histograms - OPTION 2 (after Fluctuation weighting)
        if extract_rh2:
            rh2 = np.sum(rp,axis=0) #TODO: adapt to do always without DC component
            rh2_list.append(rh2.flatten(FLATTEN_ORDER))


        # Gradient+Gauss filter
        #if extract_rp:
            # TODO Gradient+Gauss filter

            #for i in range(1,rp.shape[1]):
            #    rp[:,i-1] = np.abs(rp[:,i] - rp[:,i-1]);
            #
            #rp = blur1 * rp * blur2;

        if extract_rp:
            rp_list.append(rp.flatten(FLATTEN_ORDER))

        seg_pos = seg_pos + segment_size * step_width


    if extract_rp:
        if return_segment_features:
            features["rp"] = np.array(rp_list)
        else:
            features["rp"] = np.median(np.asarray(rp_list), axis=0)

    if extract_ssd:
        if return_segment_features:
            features["ssd"] = np.array(ssd_list)
        else:
            features["ssd"] = np.mean(np.asarray(ssd_list), axis=0)  # MEAN for SSD
        
    if extract_rh:
        if return_segment_features:
            features["rh"] = np.array(rh_list)
        else:
            features["rh"] = np.median(np.asarray(rh_list), axis=0)

    if extract_mvd:
        if return_segment_features:
            features["mvd"] = np.array(mvd_list)
        else:
            features["mvd"] = np.mean(np.asarray(mvd_list), axis=0)

    # NOTE: no return_segment_features for temporal features as they measure variation of features over time

    if extract_tssd:
        features["tssd"] = calc_statistical_features(np.asarray(ssd_list).transpose()).flatten(FLATTEN_ORDER)

    if extract_trh:
        features["trh"] = calc_statistical_features(np.asarray(rh_list).transpose()).flatten(FLATTEN_ORDER)

    if return_segment_features:
        # also include the segment positions in the result
        features["segpos"] = np.array(seg_pos_list)
        features["timepos"] = features["segpos"] / (samplerate * 1.0)

    return features


def available_feature_types():
    '''list all available feature types to be extracted by rp_extract'''
    return ['rp','ssd','rh','tssd','trh','mvd']


# function to self test rp_extract if working properly
def self_test():
    import doctest
    #doctest.testmod()
    doctest.run_docstring_examples(rp_extract, globals(), verbose=True)



if __name__ == '__main__':

    import sys
    from audiofile_read import *       # import our library for reading wav and mp3 files

    # process file given on command line or default song (included)
    if len(sys.argv) > 1:
        if sys.argv[1] == '-test': # RUN DOCSTRING SELF TEST
            print("Doing self test. If nothing is printed, it is ok.")
            import doctest
            doctest.run_docstring_examples(rp_extract, globals()) #, verbose=True)
            exit()   # Note: no output means that everything went fine
        else:
            audiofile = sys.argv[1]
    else:
        audiofile = "music/BoxCat_Games_-_10_-_Epic_Song.mp3"

    # Read audio file and extract features
    try:

        samplerate, samplewidth, wavedata = audiofile_read(audiofile)

        np.set_printoptions(suppress=True)

        bark_bands = 24  # choose the number of Bark bands (2..24)
        mod_ampl_limit = 60 # number modulation frequencies on x-axis

        feat = rp_extract(wavedata,
                          samplerate,
                          extract_rp=True,
                          extract_ssd=True,
                          extract_rh=True,
                          extract_mvd=False,
                          extract_tssd=False,
                          extract_trh=False,
                          n_bark_bands=bark_bands,
                          spectral_masking=True,
                          transform_db=True,
                          transform_phon=True,
                          transform_sone=True,
                          fluctuation_strength_weighting=True,
                          skip_leadin_fadeout=1,
                          step_width=1,
                          mod_ampl_limit=mod_ampl_limit)

        # feat is a dict containing arrays for different feature sets
        print("Successfully extracted features:" , feat.keys())

    except ValueError as e:
        print(e)
        exit()

    # example print of first extracted feature vector
    keys = feat.keys()
    k = keys[0]

    print(k.upper, " feature vector:")
    print(feat[k])

    # EXAMPLE on how to plot the features
    do_plots = False

    if do_plots:
        from rp_plot import *

        plotrp(feat["rp"],rows=bark_bands,cols=mod_ampl_limit)
        plotrh(feat["rh"])
        plotssd(feat["ssd"],rows=bark_bands)

    # EXAMPLE on how to store RP features in CSV file
    # import pandas as pd
    # filename = "features.rp.csv"
    # rp = pd.DataFrame(feat["rp"].reshape([1,feat["rp"].shape[0]]))
    # rp.to_csv(filename)
