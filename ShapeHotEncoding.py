"""
Created on Sun Jun 26 19:42:06 2016

@author: satyanar
"""
import h5py 
import numpy as np 
from collections import OrderedDict 
import os

def set_enconding():
    encoding_dictionary =  OrderedDict ( )
    encoding_dictionary['A'] = [1, 0, 0, 0]
    encoding_dictionary['C'] = [0, 1, 0, 0]
    encoding_dictionary['G'] = [0, 0, 1, 0]
    encoding_dictionary['T'] = [0, 0, 0, 1]
    encoding_dictionary['a'] = [1, 0, 0, 0]
    encoding_dictionary['c'] = [0, 1, 0, 0]
    encoding_dictionary['g'] = [0, 0, 1, 0]
    encoding_dictionary['t'] = [0, 0, 0, 1]   
    encoding_dictionary['N'] = [0, 0, 0, 0]
    encoding_dictionary['n'] = [0, 0, 0, 0]

    return encoding_dictionary
    
def revcompl(s):
    rev_s = ''.join([{'M':'g', 'g':'M', 'A':'T','C':'G','G':'C','T':'A', 'N':'N'}[B] for B in s][::-1])
    return rev_s
    

def add_RCS_with_padding (forward_strand,
                          is_padding_by_N=True, num_nuc_padding = 10, character_as_N = 'N',
                          defined_padding="CACGTC"):
    reverse_complement = revcompl (forward_strand)
    if is_padding_by_N:
        complete_strand = forward_strand + character_as_N*num_nuc_padding + reverse_complement
    else:
        complete_strand = forward_strand + defined_padding + revcompl(defined_padding) + reverse_complement
    return complete_strand

def readfasta (filename, delim=","):
        fp=open (filename, "r")
        shape_dictionary = OrderedDict()
        value = []
        last_name =""
        for line in fp:
                line = line.strip()
                if line == '>':
                        last_name = line
                else:
                        break
        value = ""
        for line in fp:
                line = line.strip()
                if line[0] == '>':
                        shape_dictionary[last_name] = value.strip(delim)
                        value = ""
                        last_name = line

                else:
                        value += delim  + line
        shape_dictionary[last_name] = value.strip(delim)                  
        fp.close()
        return shape_dictionary
    

def encode_sequence_array( sequence_array ): 
    encoded_list = []
    encoding_dictionary = set_enconding ()
    for seq in sequence_array: 
        seq_in_char_array = list (seq)
        for c in seq_in_char_array: 
            encoded_list.extend (encoding_dictionary[c])
    return encoded_list
        

def ascii_to_hd5_HotEncode(filename, add_reverse=False, num_padding=0, add_shape=True ): 
    
    filename_base, extension = os.path.splitext(filename)
    
    h5filename = filename_base + '.h5'
    sequence_array = []
    affinity_array = []
    fp = open (filename)
    for line in fp :
        line = line.strip()
        line_items= [ x for x in line.split()]
        line_items[0] = ''.join (c for c in line_items[0] if c not in 'acgt')
        if add_reverse is False:        
            sequence_array.append(line_items[0])
        else:
            sequence_array.append(add_RCS_with_padding (line_items[0], num_padding ))
        
        affinity_array.append(float(line_items[1]))
    fp.close()
    
    
    ## if add_shape is True
    mgw_array = []
    prot_array = []
    roll_array = []
    helt_array = []
    ep_array = []

    padding_mgw = []
    padding_prot = []
    padding_roll = []
    padding_helt = []
    padding_ep = []
    padding_mgw += num_padding*['0.6632835820895522'] # (5.072 - 2.85)/(6.2-2.85)
    padding_prot += num_padding*['0.5896844660194176'] # x =  -6.792 ;(float(x)-(-0.03))/(-0.03 - (-16.51))
    padding_roll += num_padding*['0.4574084834398605'] # x = -0.698 ; (float(x)-(-8.57))/(8.64-(-8.57))
    padding_helt += num_padding*['0.4762306610407878'] # x = 34.326 ; (float(x)- 30.94)/(38.05 - 30.94)
    padding_ep += num_padding*['0.8005649717514125']   # x = -6.505 ; (float(x)- (-13.59))/( -4.74 - (-13.59))

    rev_line_items = []
    if add_shape is True:
        # tranfer to fasta format and run DNAshapeR      
        # read shape files (append .fa.MGW etc)
        #fp = open (filename + '.fa.MGW')
        mgw_dict = readfasta (filename + '.fa.MGW')
        for key in mgw_dict.keys():
                line = mgw_dict[key]
                # normalizing and replacing NA with average value            
                line_items = line.strip().split(',')
                line_items =  ['5.072' if x=='NA' else x for x in line_items]
                line_items = [(float(x)-2.85)/(6.2-2.85) for x in line_items]             
                rev_line_items = line_items[::-1]
                
                mgw_array.extend(line_items)
                mgw_array.extend(padding_mgw)
                mgw_array.extend(rev_line_items)
       
        #fp.close () 
        #fp = open (filename + '.fa.ProT')
        prot_dict = readfasta(filename + '.fa.ProT')
        for key in prot_dict.keys():
                line = prot_dict[key]
                # normalizing and replacing NA with average value            
                line_items = line.strip().split(',')
                line_items =  ['-6.792' if x=='NA' else x for x in line_items]
                line_items = [(float(x)-(-16.51))/(-0.03 - (-16.51)) for x in line_items]             
                rev_line_items = line_items[::-1]
                
                prot_array.extend(line_items)
                prot_array.extend(padding_prot)
                prot_array.extend(rev_line_items)
       
              
        #fp.close () 
        #fp = open (filename + '.fa.Roll')
        roll_dict = readfasta(filename + '.fa.Roll')
        for key in roll_dict.keys():
                line = roll_dict[key]
                # normalizing and replacing NA with average value            
                line_items = line.strip().split(',')
                line_items =  ['-0.698' if x=='NA' else x for x in line_items]
                line_items.append ('-0.698')
                line_items = [(float(x)-(-8.57))/(8.64-(-8.57)) for x in line_items]             
                rev_line_items = line_items[::-1]

                roll_array.extend(line_items)
                roll_array.extend(padding_roll)
                roll_array.extend(rev_line_items)
                #roll_array.extend([-0.698, -0.698])                
        
        #fp.close () 
        #fp = open (filename + '.fa.HelT')
        helt_dict = readfasta(filename + '.fa.HelT')
        for key in helt_dict.keys():
                line = helt_dict[key]
                # normalizing and replacing NA with average value            
                line_items = line.strip().split(',')
                line_items =  ['34.326' if x=='NA' else x for x in line_items]
                line_items.append ('34.326')
                line_items = [(float(x)- 30.94)/(38.05 - 30.94) for x in line_items]             
                rev_line_items = line_items[::-1]
                
                helt_array.extend(line_items)
                helt_array.extend(padding_helt)
                helt_array.extend(rev_line_items)
                #helt_array.extend([34.326, 34.326])
                      
        #fp.close () 
        #fp = open (filename + '.fa.EP')
        ep_dict = readfasta(filename + '.fa.EP')
        for key in ep_dict.keys():
                line = ep_dict[key]
                # normalizing and replacing NA with average value            
                line_items = line.strip().split(',')
                line_items =  ['0.8005649717514125' if x=='NA' else x for x in line_items]
                line_items = [(float(x)- (-13.59))/( -4.74 - (-13.59)) for x in line_items]             
                rev_line_items = line_items[::-1]
                
                ep_array.extend(line_items)
                ep_array.extend(padding_ep)
                ep_array.extend(rev_line_items)
                      
    #print len(mgw_array)
    #print len (sequence_array)
    num_rows  = len (sequence_array)
    length_of_one_sequence = len (sequence_array[0])
    fp_hdf = h5py.File (h5filename, "w")
    dgroup = fp_hdf.create_group("data")
    dtarget = fp_hdf.create_group("targets")
    
    #dset = fp_hdf.create_dataset("sequence ", (5,), dtype='a16')
    #dset[...] = sequence_array[0:5]
    #vset = fp_hdf.create_dataset("bit_strings", (5,2,4), dtype = 'i1')
    #vset[...] = np.zeros(40, dtype='i1').reshape(5,2,4)
    #
    #print vset 
    #fp_hdf.close()
    dt = h5py.special_dtype(vlen=unicode) 
    dset = dgroup.create_dataset("sequence", (num_rows,), dtype=dt)
    dset[...] = sequence_array
    vset = dgroup.create_dataset("s_x", (num_rows,length_of_one_sequence,4), dtype = 'i1')
    vset_seqen = dgroup.create_dataset("seqen_x", (num_rows,length_of_one_sequence,4), dtype = 'i1')
    
    encoded_seq_array = encode_sequence_array(sequence_array) 
        
    #print encoded_seq_array[0]
    print len (encoded_seq_array)
    #print "---------------"
    np_encoded_seq_array = np.asarray (encoded_seq_array, dtype = 'i1') 
    #vset[...] = np.zeros(40, dtype='i1').reshape(5,2,4)
    #v= np_encoded_seq_array.reshape(num_rows,length_of_one_sequence,4)
    sequence_HotEncode = np_encoded_seq_array.reshape(num_rows, length_of_one_sequence, 4)
    sequence_HotEncode = sequence_HotEncode.transpose (0,2,1)

    vset[...] = np_encoded_seq_array.reshape(num_rows,length_of_one_sequence,4)

    vset_seqen[...] =np_encoded_seq_array.reshape(num_rows,length_of_one_sequence,4)
    
    # add mgw
    if add_shape is True:
        print "-------------##### -------------\n"
        print len(mgw_array), "\n"
        print "-------------##### -------------\n"
        print length_of_one_sequence , "\n"
        mgwset = dgroup.create_dataset("mgw_x", (num_rows, length_of_one_sequence, 1), dtype = 'f')
        np_mgw = np.asarray(mgw_array, dtype = 'f')
        arr = np_mgw.reshape(num_rows, length_of_one_sequence, 1)
        mgw_HotEncode = np_mgw.reshape(num_rows, length_of_one_sequence)
        #np.random.shuffle(arr)    #only shuffles the first dim.
        mgwset [...] = arr
#        mgwset = dgroup.create_dataset("mgw_x", (num_rows,length_of_one_sequence,4), dtype = 'i1')
#        #np_mgw = np.asarray(mgw_array, dtypv e = 'f')
#        mgwset [...] = np_encoded_seq_array.reshape(num_rows,length_of_one_sequence,4)
        # add prot 
        protset = dgroup.create_dataset("prot_x", (num_rows, length_of_one_sequence, 1), dtype = 'f')
        np_prot = np.asarray(prot_array, dtype = 'f')
        arr = np_prot.reshape(num_rows, length_of_one_sequence, 1) 
        prot_HotEncode = np_prot.reshape(num_rows, length_of_one_sequence)
        #np.random.shuffle(arr)    #only shuffles the first dim.
        protset [...] = arr
        
        # add roll 
        rollset = dgroup.create_dataset("roll_x", (num_rows, length_of_one_sequence, 1), dtype = 'f')
        np_roll = np.asarray(roll_array, dtype = 'f')
        arr = np_roll.reshape(num_rows, length_of_one_sequence, 1)
        roll_HotEncode = np_roll.reshape(num_rows, length_of_one_sequence)
        #np.random.shuffle(arr)    #only shuffles the first dim.
        rollset [...] = arr

        # add helt 
        heltset = dgroup.create_dataset("helt_x", (num_rows, length_of_one_sequence, 1), dtype = 'f')
        np_helt = np.asarray(helt_array, dtype = 'f')
        arr = np_helt.reshape(num_rows, length_of_one_sequence, 1)
        helt_HotEncode = np_helt.reshape(num_rows, length_of_one_sequence)
        #np.random.shuffle(arr)    #only shuffles the first dim.
        heltset [...] = arr
        
        # add ep
        epset = dgroup.create_dataset("ep_x", (num_rows, length_of_one_sequence, 1), dtype = 'f')
        np_ep = np.asarray(ep_array, dtype = 'f')
        arr = np_ep.reshape(num_rows, length_of_one_sequence, 1)
        ep_HotEncode = np_ep.reshape(num_rows, length_of_one_sequence)
        #np.random.shuffle(arr)    #only shuffles the first dim.
        epset [...] = arr 

        finalShapeMatrix = np.concatenate ((mgw_HotEncode, 
                                            prot_HotEncode, 
                                            roll_HotEncode, 
                                            helt_HotEncode), axis = 1)
        
        finalShapeMatrix = finalShapeMatrix.reshape(num_rows, 1, 4, length_of_one_sequence)
        print (finalShapeMatrix.shape)
    
    print "----------------\n"
    print (affinity_array[1:5])
    print ("----------------\n")
    yset = dgroup.create_dataset("c0_y", (num_rows,1), dtype = 'f') 

    np_affinity =np.asarray(affinity_array, dtype = 'f') 
    print (np_affinity[1:5])    
    np_affinity_bool = [True if x > 0.6 else False for x in  np_affinity] 
    np_affinity_bool = np.asarray(np_affinity_bool, dtype = 'bool')
    print (np_affinity_bool[1:5])
    affinity_HotEncode = np_affinity_bool.reshape (num_rows,1)
  

    yset [...] = np_affinity.reshape(num_rows,1)
    print len (np_affinity)
    df = dtarget.create_dataset("name", (1,), dtype='a16' )
    targetname = ["dummy"]
    df[...] = targetname 
    
    df1 = dtarget.create_dataset("id", (1,), dtype='a16' )
    targetname = ["c0"]
    df1[...] = targetname    
    
    #np.asarray(targetname, dtype='a16')
    print "h5 file is written to " + h5filename 
    fp_hdf.close()
    #return (h5filename, finalShapeMatrix, affinity_HotEncode) 
#    kk = 5
#    GG = 7
    return (sequence_HotEncode, affinity_HotEncode)
    #return (kk, GG)


