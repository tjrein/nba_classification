==== CS613 ====
    Author: Tom Rein
    Email: tr557@drexel.edu

==== Dependencies ====
    * python3
    * pip3
    * numpy
    * scikit-learn
    * matplotlib
    * pandas

==== Files Included ====
    * cluster_lda.py
    * cluster_pca.py
    * cluster_pca_plus_lda.py
    * data.py
    * get_statistical_profile.py
    * merge_csv.py
    * query_players.py
    * reduce_dimensions.py


==== cluster_lda.py ====
    Script used to show the GMM clustering process after LDA projection.
    Displays two scatter plots:
        * Data prior to clustering with colors corresponding to positions.  
        * Data after clustering.

    To execute:
        > python3 cluster_lda.py


==== cluster_pca.py ====
    Script used to show the GMM clustering process after PCA projection.
    Displays two scatter plots:
        * Data prior to clustering with colors corresponding to positions.  
        * Data after clustering.

    To execute:
        > python3 cluster_pca.py


==== cluster_pca_plus_lda.py ====
    Script used to show the GMM clustering process after PCA plus LDA projection.
    Displays two scatter plots:
        * Data prior to clustering with colors corresponding to positions.  
        * Data after clustering.

    To execute:
        > python3 cluster_pca_plus_lda.py


==== data.py ====
    Contains functions used for processing dataset. Used by various scripts
    This file is configured to read from './data/nba_2017_combined.csv'
    Other combined .csv files from different NBA seasons can also be used by editing this file.


==== get_statistical_profile.py ====
    Script used to display statistical profiles for computed clusters.
    Displays a bar graph containing statistical averages of group compared with league averages.

    Can accept two command line args: method and group
    method can be one of 'pca', 'lda', or '+'
    group corresponds to a cluster number.
    method is always the first argument and group is always the second argument

    To execute:
        > python3 get_statistical_profile.py {method} {group}
    
    ex. python3 get_statistical_profile.py lda 6

    If this script is called without arguments, it will output all the groups and players clustered in those groups for a given method.
    Calling this script without arguments can be used to determine which group numbers are present in a given method. 
    The default method is PCA.


==== merge_csv.py ====
    This script is intended to merge the pergame statistics .csv file with the advanced statistics .csv file to produce a combine statistics .csv file.
    This scipt is configured to work with the 2017 NBA statistics, but this can be modified in the file to work with any season from basketball-reference

    To execute:
        > python3 merge_csv.py


==== query_players.py ====  
    This script can be used to examine the probablities of a player belonging to various clusters.
    
    Can accept two command line arguments: method and player.
    method is one of 'PCA', 'LDA', or '+'
    player is a player's full name
    method is always the first argument and player is always the second argument

    To execute:
        > python3 query_players.py {method} {player}

    ex. python3 query_players.py + 'hassan whiteside'

    If this script is executed without a player argument, it will display probabilities for all players in alphabetical order.
    If a method is not specified, PCA will be the default.


==== reduce_dimensions.py === 
    This file contains functions used by other scripts for dimensionality reduction and clustering.






