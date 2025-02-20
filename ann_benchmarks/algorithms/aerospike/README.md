# Backoff Logic when encountering Resource Exhausted

This only applies when "ignoreExhaustedEvent" is true in the config.yml (default is false). When this value is True, any exhausted resource event will be handled by the healer.

# The “back-off” logic is as follows:

-   When the exception is received:
    -   First record in error will perform the following actions:
        -   Signal the “main” populating task to go into “sleep mode” so that additional records will not be upserted.
        -   A warning message is logged.
    -   All records in error will do the following:
        -   Call “wait for index completion.”
        -   Once the index is built the following occurs:
            -   Re-upsert the error records
            -   If successful, signal the “main” populating task to re-start populating.
            -   A warning message is logged, stating population has re-started.

# Environmental Variables

Below are the Environmental Variables:

-   AVS_LOGLEVEL -- The Vector Client API's Log Level. Defaults "WARNING"
    -   Possible Values:
    -   CRITICAL
    -   FATAL
    -   ERROR
    -   WARNING
    -   WARN
    -   INFO
    -   DEBUG
    -   NOTSET

        Note: The logging file is determined by "APP_LOGFILE"

-   AVS_HOST -- The AVS Server's Address. Defaults to "localhost"
-   AVS_PORT -- The AVS Server's Port. Defaults to 5000
-   AVS_USELOADBALANCER -- The AVS Server's Address is a Load Balancer. Default False.
-   AVS_NAMESPACE -- The Vector's Namespace. Defaults to "test"
-   AVS_SET -- The Vector's Set name. Defaults to "ANN-data"

    This behavior is determined by the "uniqueSetIdxName" argument defined in the config.yml file.

    The default (True) behavior is to create a unique Set name where this is the prefix to that name.

The name has the following parts:

```
{AVS_SET}_{ANN Distance Type}_{AVS Idx Type}_{Dimension}_{hnsw m}_{hnsw ef construction}_{hnsw ef}
```

Example:

ANN-data_angular_COSINE_20_16_100_100

If "uniqueSetIdxName" is false, the Set name is as follows:

```
{AVS_SET}__{ANN Distance Type}_{AVS Idx Type}
```

Example:

ANN-data_angular_COSINE

-   APP_LOGFILE -- The Aerospike's ANN Logging file. Default is "AerospikeANN.log".

    The folder is always the current working directory.

-   APP_LOGLEVEL -- The Aerospike's ANN Log Level. Defaults "INFO"
    -   Possible Values:
    -   CRITICAL
    -   FATAL
    -   ERROR
    -   WARNING
    -   WARN
    -   INFO
    -   DEBUG
    -   NOTSET

Note: For performance testing this should be set to "NOTSET".

When running in a docker container, logging is disabled.

-   APP_DROP_IDX -- A Boolean value that will determine if the Vector index is dropped if it already exists. The default is to use "dropIdx" argument in the config.yml file.
-   APP_INDEX_SLEEP -- The amount of time to sleep after the index is dropped. The default is 0.

    Possible values are:

    -   0 -- Don't Sleep
    -   \< 0 -- The number of seconds to sleep
-   APP_POPULATE_TASKS -- The number of concurrent records upserted (put) tasks that are performed during the index population phase. When this number of records are upserted, the app will wait until all upserts are completed and then process the next set of records. The default is 5000.

    Values:

    -   \< 0 -- All records are upserted, concurrently, and the app will only wait for the upsert completion before waiting for index completion.
    -   0 or 1 -- One record is upserted at a time (sync)
    -   \> 1 -- The number of records upserted, concurrently (async), before the app waits for the upserts to complete.
-   APP_PINGAVS -- Checks to determine if the AVS server is reachable via ping. Default is False.
-   APP_CHECKRESULT -- Checks the Vector Search results for failed results or Zero Distance. Default is True

    Note: This value is always false if running in a docker container.

    This should be set to False when conducting performance testing!

The default bin name for the vectors is always "ANN_embedding".

# config.yml file

Using the config.yml file. The Aerospike ANN config.yml file can support the different ANN run group configurations. It is suggested that the Aerospike ANN application is ran using the ANN Distance Type configuration.

Using this configuration, we can match each ANN distance type (i.e., Angular, Euclidean, Jaccard, etc.) to the "best" Aerospike Vector index type. Below is an example of this configuration with comments regarding the behavior of each parameter:

```
float:
#This defines a run group based on the ANN angular datasets.
  angular:
#All entries to “run_groups” keyword are required as-is (cannot change the values or structure)!
  - base_args: ['@metric',  '@dimension']
    constructor: Aerospike
    disabled: false #can change to true to disable this run-group
    docker_tag: ann-benchmarks-aerospike
    module: ann_benchmarks.algorithms.aerospike
    name: aerospike
    run_groups:
      cosine: #Should match Idx Type
#This grouping is reqired
        args: [
          [cosine], #Idx Type, any Aerospike Index Type, case insensitive). This is required…
#A collection of HnswParams where each param is ran as a separate ran for this Idx Type. This is required and must have at least one item.
          [{m: 8, ef_construction: 64, ef: 8},
            {m: 16, ef_construction: 128, ef: 8} ],
#Unique Set/Index Name (optional, default True). See the “AVS_SET” environment variable above.
          [True],
#True to Drop Idx and Re-Populate, optional default true. See “APP_DROP_IDX” environment variable above.
          [True],
#Determines what phases are executed. Values are:
#	IdxPopulateOnly – only conduct the populate index phase,
#	QueryOnly – only perform the vector search phase,
#	AllOps – All phases (optional default value)
          [AllOps]
        ]
#This grouping is required
        query_args: [
# If provided (optional), overrides the HnswParams defined above for the vector search phase
          [null, #Uses default defined above
            {ef: 10} #Override “ef” above
		]
      ]
#This defines another run group based on the ANN Euclidean datasets.
#This show using the required params.
  euclidean:
  - base_args: ['@metric',  '@dimension']
    constructor: Aerospike
    disabled: false
    docker_tag: ann-benchmarks-aerospike
    module: ann_benchmarks.algorithms.aerospike
    name: aerospike
    run_groups:
      SQUARED_EUCLIDEAN:
              args: [
                [SQUARED_EUCLIDEAN], #Idx Type
                [{m: 16, ef_construction: 100, ef: 100}]
              ]
              query_args: [
                []
              ]
```

# HDF5 Dataset Additional Attributes

The following attributes are added in the resulting ANN HDF5 dataset (note that all added attributes are prefixed with "as_"):

-   as_indockercontainer – True if this run was within a docker container. False if it was ran natively
-   as_idx_name – The name of the index
-   as_idx_type – The Aerospike Vector Index Type
-   as_idx_binname – The Index’s Bin name
-   as_idx_hnswparams – The index’s “hnsw” parameters as passed into this run via the config file. Any missing or None values will use the default values defined by Aerospike Vector client/server.
-   as_idx_drop – True if the index will be dropped
-   as_idx_ignoreexhuseevents – True to ignore any “Exhausted Resource” errors and the Aerospike Vector Healer will be used to reconcile the index. If false, internal “back-off” logic is ued.
-   as_idx_definition_built - Only available when the database is populated. The actual Vector Index's definitions with default values.
-   as_actions – The actions performed in this run (e.g., All actions, Populate Index Only, Query Only, etc.)
-   as_host – The Aerospike Vector server
-   as_isloadbalancer – If present, the as_host is a load balancer
-   as_namespace – The Aerospike Namespace used for the tun
-   as_set – The Aerospike Set name used for the run
-   as_train_shape - The dimensions of the training dataset which is used to populate the database.
-   as_query_hnswsearchparams – The Query’s “hnsw” parameters as passed into this run via the config file. Any missing or None values will use the default values defined by Aerospike Vector client/server.
-   as_query_checkresults – If true the query results are checked/validated. This should be false for timing runs.
-   as_query_no_result_cnt - The number of queries that returned empty results. Only available if the query check results are true.
-   as_query_no_neighbors_fnd - The number of queries that returned no neighbors. Only available if the query check results are true.
-   as_upserted_vectors – The number of vectors inserted
-   as_upserted_time_secs – The amount of time to perform all the inserts in seconds. This doesn’t include index build completion.
-   as_idx_completion_secs – The number of seconds to complete the index build. Does not include inset time.
-   as_total_polulation_time_secs – The complete time to insert and build the index.

# Installation

## Docker

Need to install docker and the associated python package. Below are the instructions for [Ubuntu](https://docs.docker.com/engine/install/ubuntu/):

1.  \# Add Docker's official GPG key:
    1.  sudo apt-get update
    2.  sudo apt-get install ca-certificates curl
    3.  sudo install -m 0755 -d /etc/apt/keyrings
    4.  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    5.  sudo chmod a+r /etc/apt/keyrings/docker.asc
2.  \# Add the repository to Apt sources:
    1.  echo "deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linu\> \$(. /etc/os-release && echo "\${UBUNTU_CODENAME:-\$VERSION_CODENAME}") stable" \| sudo tee /etc/apt/sources.list.d/doc
3.  sudo apt-get update
4.  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
5.  sudo docker run hello-world
6.  pip3 install docker

# Python Packages

In the “aerospike” folder under “algorithms” folder:

1.  pip3 install -r requirements.txt
