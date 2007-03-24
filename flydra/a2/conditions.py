##2/3 height post experiments:

##                             odor
##                    yes                no

##         yes     20070305           20070304  
##                 20070306           20070308
##wind       
##         no      20070303           20070126
##                 20070307           20070127
##                                    20070128
##                                    20070129

# this is some kind of order
##condition_names = ['no post','tall post','spot',
##                   'necklace','half post','double post']

condition_names = ['half post',
                   'half no odor, w/ wind',
                   'half w/ odor',
                   'half w/ odor, w/ wind',
                   ]

stim_names = {'tall post':'tall',
              'no post':None,
              'spot':None,
              'necklace':'necklace',
              'double post':'double',
              'd2':'double_20070301',
              
              'half post':'half',
              'half w/ odor':'half_20070303',
              'half w/ odor, w/ wind':'half_20070303',
              'half no odor, w/ wind':'half_20070303', # XXX need to double check using ukine
              
              }

files = {
##    'half post':['DATA20070126_184022.h5',
##                 'DATA20070127_165515.h5',
##                 'DATA20070128_171253.h5',
##                 'DATA20070129_184131.h5',
##                 ],
    'half post':['DATA20070126_184022_smoothed.mat',
                 'DATA20070127_165515_smoothed.mat',
                 'DATA20070128_171253_smoothed.mat',
                 'DATA20070129_184131_smoothed.mat',
                 ],
    
    'half no odor, w/ wind':['DATA20070304_185411_smoothed.mat',
                             'DATA20070308_185556_smoothed.mat',
                             ],
    
    'half w/ odor':['DATA20070303_191938_smoothed.mat',
                    'DATA20070307_190812_smoothed.mat',
                    ],
    
    'half w/ odor, w/ wind':['DATA20070305_181659_smoothed.mat',
                             'DATA20070306_190445_smoothed.mat',
                             ],
    
    #######################################################
    
    'd2':['DATA20070301_155517.h5'],
    
    
    'tall post':['DATA20061206_192530.kalmanized.h5',
                 'DATA20061207_183409.kalmanized.h5',
                 'DATA20061208_181556.kalmanized.h5',
                 ],
    'no post':['DATA20061209_180630.kalmanized.h5',
               'DATA20061215_174134.kalmanized.h5',
               'DATA20061218_180311.kalmanized.h5',
               'DATA20061223_173845.kalmanized.h5',
               ],
    'spot':['DATA20061211_183352.kalmanized.h5',
            'DATA20061212_184958.kalmanized.h5',
            'DATA20061213_181940.kalmanized.h5',
            ],
    'necklace':['DATA20061219_184831.kalmanized.h5',
                'DATA20061220_184522.kalmanized.h5',
                'DATA20061221_184519.kalmanized.h5',
                'DATA20061222_173500.kalmanized.h5',
                ],
    'double post':['DATA20070130_184845.h5',
                   'DATA20070201_190332.h5',
                   'DATA20070202_190006.h5',
                   #'DATA20070202_190006.kalmanized.h5', # recovered 2D data used
                   ],
    }
