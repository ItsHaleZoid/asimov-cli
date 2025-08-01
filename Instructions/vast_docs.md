Website logo
Console
Discord
Search or ask...
⌘
K
CLI
Commands
72 min
CLI Commands
usage: vastai [-h] [--url URL] [--retry RETRY] [--raw] [--explain] [--curl] [--api-key API_KEY] [--version] command ...

positional arguments:
 command               command to run. one of:
  help                 print this help message
  attach ssh           Attach an ssh key to an instance. This will allow you to connect to the instance with the ssh key
  cancel copy          Cancel a remote copy in progress, specified by DST id
  cancel sync          Cancel a remote copy in progress, specified by DST id
  change bid           Change the bid price for a spot/interruptible instance
  clone volume         Clone an existing volume
  copy                 Copy directories between instances and/or local
  cloud copy           Copy files/folders to and from cloud providers
  take snapshot        Schedule a snapshot of a running container and push it to your repo in a container registry
  create api-key       Create a new api-key with restricted permissions. Can be sent to other users and teammates
  create env-var       Create a new user environment variable
  create ssh-key       Create a new ssh-key
  create autogroup     Create a new autoscale group
  create endpoint      Create a new endpoint group
  create instance      Create a new instance
  create subaccount    Create a subaccount
  create team          Create a new team
  create team-role     Add a new role to your team
  create template      Create a new template
  create volume        Create a new volume
  delete api-key       Remove an api-key
  delete ssh-key       Remove an ssh-key
  delete scheduled-job
                       Delete a scheduled job
  delete autogroup     Delete an autogroup group
  delete endpoint      Delete an endpoint group
  delete env-var       Delete a user environment variable
  delete template      Delete a Template
  delete volume        Delete a volume
  destroy instance     Destroy an instance (irreversible, deletes data)
  destroy instances    Destroy a list of instances (irreversible, deletes data)
  destroy team         Destroy your team
  detach ssh           Detach an ssh key from an instance
  execute              Execute a (constrained) remote command on a machine
  get endpt-logs       Fetch logs for a specific serverless endpoint group
  invite member        Invite a team member
  label instance       Assign a string label to an instance
  launch instance      Launch the top instance from the search offers based on the given parameters
  logs                 Get the logs for an instance
  prepay instance      Deposit credits into reserved instance
  reboot instance      Reboot (stop/start) an instance
  recycle instance     Recycle (destroy/create) an instance
  remove member        Remove a team member
  remove team-role     Remove a role from your team
  reports              Get the user reports for a given machine
  reset api-key        Reset your api-key (get new key from website)
  start instance       Start a stopped instance
  start instances      Start a list of instances
  stop instance        Stop a running instance
  stop instances       Stop a list of instances
  search benchmarks    Search for benchmark results using custom query
  search invoices      Search for benchmark results using custom query
  search offers        Search for instance types using custom query
  search templates     Search for template results using custom query
  search volumes       Search for volume offers using custom query
  set api-key          Set api-key (get your api-key from the console/CLI)
  set user             Update user data from json file
  ssh-url              ssh url helper
  scp-url              scp url helper
  show api-key         Show an api-key
  show api-keys        List your api-keys associated with your account
  show audit-logs      Display account's history of important actions
  show scheduled-jobs  Display the list of scheduled jobs
  show ssh-keys        List your ssh keys associated with your account
  show autogroups      Display user's current autogroup groups
  show endpoints       Display user's current endpoint groups
  show connections     Display user's cloud connections
  show deposit         Display reserve deposit info for an instance
  show earnings        Get machine earning history reports
  show env-vars        Show user environment variables
  show invoices        Get billing history reports
  show instance        Display user's current instances
  show instances       Display user's current instances
  show ipaddrs         Display user's history of ip addresses
  show user            Get current user data
  show subaccounts     Get current subaccounts
  show members         Show your team members
  show team-role       Show your team role
  show team-roles      Show roles for a team
  show volumes         Show stats on owned volumes.
  create cluster       Create Vast cluster
  join cluster         Join Machine to Cluster
  delete cluster       Delete Cluster
  remove-machine-from-cluster  Removes machine from cluster
  show overlays        Show overlays associated with your account.
  create overlay       Creates overlay network on top of a physical cluster
  join overlay         Adds instance to an overlay network
  delete overlay       Deletes overlay and removes all of its associated instances
  show clusters        Show clusters associated with your account.
  transfer credit      Transfer credits to another account
  update autogroup     Update an existing autoscale group
  update endpoint      Update an existing endpoint group
  update env-var       Update an existing user environment variable
  update instance      Update recreate an instance from a new/updated template
  update team-role     Update an existing team role
  update template      Update an existing template
  update ssh-key       Update an existing ssh key
  cancel maint         [Host] Cancel maint window
  cleanup machine      [Host] Remove all expired storage instances from the machine, freeing up space
  delete machine       [Host] Delete machine if the machine is not being used by clients. host jobs on their own machines are disregarded and machine is force deleted.
  list machine         [Host] list a machine for rent
  list machines        [Host] list machines for rent
  list volume          [Host] list disk space for rent as a volume on a machine
  list volumes         [Host] list disk space for rent as a volume on machines
  unlist volume        [Host] unlist volume offer
  remove defjob        [Host] Delete default jobs
  set defjob           [Host] Create default jobs for a machine
  set min-bid          [Host] Set the minimum bid/rental price for a machine
  schedule maint       [Host] Schedule upcoming maint window
  show machine         [Host] Show hosted machines
  show machines        [Host] Show hosted machines
  show maints          [Host] Show maintenance information for host machines
  unlist machine       [Host] Unlist a listed machine
  self-test machine    [Host] Perform a self-test on the specified machine

options:
 -h, --help            show this help message and exit
 --url URL             server REST api url
 --retry RETRY         retry limit
 --raw                 output machine-readable json
 --explain             output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl                show a curl equivalency to the call
 --api-key API_KEY     api key. defaults to using the one stored in /home/scott_vast/.config/vastai/vast_api_key
 --version             show version

Use 'vast COMMAND --help' for more info about a command

﻿
Client Commands
cancel copy
Cancel a remote copy in progress, specified by DST id

usage: vastai cancel copy DST

positional arguments:
  dst                instance_id:/path to target of copy operation.

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Use this command to cancel any/all current remote copy operations copying to a specific named instance, given by DST.
Examples:
 vastai cancel copy 12371

The first example cancels all copy operations currently copying data into instance 12371

﻿
cancel sync
Cancel a remote copy in progress, specified by DST id

usage: vastai cancel sync DST

positional arguments:
  dst                instance_id:/path to target of sync operation.

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Use this command to cancel any/all current remote cloud sync operations copying to a specific named instance, given by DST.
Examples:
 vastai cancel sync 12371

The first example cancels all copy operations currently copying data into instance 12371

﻿
change bid
Change the bid price for a spot/interruptible instance

usage: vastai change bid id [--price PRICE]

positional arguments:
  id                 id of instance type to change bid

options:
  -h, --help         show this help message and exit
  --price PRICE      per machine bid price in $/hour
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Change the current bid price of instance id to PRICE.
If PRICE is not specified, then a winning bid price is used as the default.

﻿
cloud copy
Copy files/folders to and from cloud providers

usage: vastai cloud_copy SRC DST CLOUD_SERVICE INSTANCE_ID CLOUD_SERVICE_SELECTED TRANSFER

options:
  -h, --help            show this help message and exit
  --src SRC             path to source of object to copy.
  --dst DST             path to target of copy operation.
  --instance INSTANCE   id of the instance
  --connection CONNECTION
                        id of cloud connection on your account
  --transfer TRANSFER   type of transfer, possible options include Instance To
                        Cloud and Cloud To Instance
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

Copies a directory from a source location to a target location. Each of source and destination
directories can be either local or remote, subject to appropriate read and write
permissions required to carry out the action. The format for both src and dst is [instance_id:]path.
You can find more information about the cloud copy operation here: https://docs.vast.ai/instances/cloud-sync

Examples:
 vastai cloud_copy --src folder --dst /workspace --cloud_service "Amazon S3" --instance_id 6003036 --cloud_service_selected 52 --transfer "Instance To Cloud"

The example copies all contents of /folder into /workspace on instance 6003036 from Amazon S3.

﻿
copy
Copy directories between instances and/or local

usage: vastai copy SRC DST

positional arguments:
  src                   instance_id:/path to source of object to copy.
  dst                   instance_id:/path to target of copy operation.

options:
  -h, --help            show this help message and exit
  -i IDENTITY, --identity IDENTITY
                        Location of ssh private key
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

Copies a directory from a source location to a target location. Each of source and destination
directories can be either local or remote, subject to appropriate read and write
permissions required to carry out the action. The format for both src and dst is [instance_id:]path.

You should not copy to /root or / as a destination directory, as this can mess up the permissions on your instance ssh folder, breaking future copy operations (as they use ssh authentication)
You can see more information about constraints here: https://docs.vast.ai/instances/data-movement

Examples:
 vastai copy 6003036:/workspace/ 6003038:/workspace/
 vastai copy 11824:/data/test data/test
 vastai copy data/test 11824:/data/test

The first example copy syncs all files from the absolute directory '/workspace' on instance 6003036 to the directory '/workspace' on instance 6003038.
The second example copy syncs the relative directory 'data/test' on the local machine from '/data/test' in instance 11824.
The third example copy syncs the directory '/data/test' in instance 11824 from the relative directory 'data/test' on the local machine.

﻿
create api-key
Create a new api-key with restricted permissions using the format found on ﻿ 

usage: vastai create api-key

options:
  -h, --help            show this help message and exit
  --permissions PERMISSIONS
                        file path for json encoded permissions, look in the
                        docs for more information
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

﻿
create autoscaler
Create a new autoscale group

usage: vastai autoscaler create [OPTIONS]

options:
  -h, --help            show this help message and exit
  --min_load MIN_LOAD   minimum floor load in perf units/s (token/s for LLms)
  --target_util TARGET_UTIL
                        target capacity utilization (fraction, max 1.0,
                        default 0.9)
  --cold_mult COLD_MULT
                        cold/stopped instance capacity target as multiple of
                        hot capacity target (default 2.5)
  --gpu_ram GPU_RAM     estimated GPU RAM req (independent of search string)
  --template_hash TEMPLATE_HASH
                        template hash (optional)
  --template_id TEMPLATE_ID
                        template id (optional)
  --search_params SEARCH_PARAMS
                        search param string for search offers ex: "gpu_ram>=23
                        num_gpus=2 gpu_name=RTX_4090 inet_down>200
                        direct_port_count>2 disk_space>=64"
  --launch_args LAUNCH_ARGS
                        launch args string for create instance ex: "--onstart
                        onstart_wget.sh --env '-e ONSTART_PATH=https://s3.amaz
                        onaws.com/vast.ai/onstart_OOBA.sh' --image
                        atinoda/text-generation-webui:default-nightly --disk
                        64"
  --endpoint_name ENDPOINT_NAME
                        deployment endpoint name (allows multiple autoscale
                        groups to share same deployment endpoint)
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

Create a new autoscaling group to manage a pool of worker instances.

Example: vastai create autoscaler --min_load 100 --target_util 0.9 --cold_mult 2.0 --search_params "gpu_ram>=23 num_gpus=2 gpu_name=RTX_4090 inet_down>200 direct_port_count>2 disk_space>=64" --launch_args "--onstart onstart_wget.sh  --env '-e ONSTART_PATH=https://s3.amazonaws.com/vast.ai/onstart_OOBA.sh' --image atinoda/text-generation-webui:default-nightly --disk 64" --gpu_ram 32.0 --endpoint_name "LLama"

﻿
create instance
Create a new instance

usage: vastai create instance ID [OPTIONS] [--args ...]

positional arguments:
  ID                    id of instance type to launch (returned from search
                        offers)

options:
  -h, --help            show this help message and exit
  --price PRICE         per machine bid price in $/hour
  --disk DISK           size of local disk partition in GB
  --image IMAGE         docker container image to launch
  --login LOGIN         docker login arguments for private repo
                        authentication, surround with ''
  --label LABEL         label to set on the instance
  --onstart ONSTART     filename to use as onstart script
  --onstart-cmd ONSTART_CMD
                        contents of onstart script as single argument
  --entrypoint ENTRYPOINT
                        override entrypoint for args launch instance
  --ssh                 Launch as an ssh instance type.
  --jupyter             Launch as a jupyter instance instead of an ssh
                        instance.
  --direct              Use (faster) direct connections for jupyter & ssh.
  --jupyter-dir JUPYTER_DIR
                        For runtype 'jupyter', directory in instance to use to
                        launch jupyter. Defaults to image's working directory.
  --jupyter-lab         For runtype 'jupyter', Launch instance with jupyter
                        lab.
  --lang-utf8           Workaround for images with locale problems: install
                        and generate locales before instance launch, and set
                        locale to C.UTF-8.
  --python-utf8         Workaround for images with locale problems: set
                        python's locale to C.UTF-8.
  --env ENV             env variables and port mapping options, surround with
                        ''
  --args ...            list of arguments passed to container ENTRYPOINT.
                        Onstart is recommended for this purpose.
  --create-from CREATE_FROM
                        Existing instance id to use as basis for new instance.
                        Instance configuration should usually be identical, as
                        only the difference from the base image is copied.
  --force               Skip sanity checks when creating from an existing
                        instance
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

Performs the same action as pressing the "RENT" button on the website at https://console.vast.ai/create/
Creates an instance from an offer ID (which is returned from "search offers"). Each offer ID can only be used to create one instance.
Besides the offer ID, you must pass in an '--image' argument as a minimum.

Examples:
vastai create instance 6995713 --image pytorch/pytorch --disk 40 --env '-p 8081:80801/udp -h billybob' --ssh --direct --onstart-cmd "env | grep _ >> /etc/environment; echo 'starting up'";
vastai create instance 384827  --image bobsrepo/pytorch:latest --login '-u bob -p 9d8df!fd89ufZ docker.io' --jupyter --direct --env '-e TZ=PDT -e XNAME=XX4 -p 22:22 -p 8080:8080' --disk 20

Return value:
Returns a json reporting the instance ID of the newly created instance.
Example: {'success': True, 'new_contract': 7835610}

﻿
create overlay
Create an overlay network inside a physical cluster. 

﻿

usage: vastai create overlay CLUSTER_ID OVERLAY_NAME

positional arguments:
 cluster_id         ID of cluster to create overlay on top of
 name               overlay network name

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key
 --version          show version

Creates an overlay network to allow local networking between instances on a physical cluster

﻿
create subaccount
Create a subaccount

usage: vastai create subaccount --email EMAIL --username USERNAME --password PASSWORD --type TYPE

options:
  -h, --help           show this help message and exit
  --email EMAIL        email address to use for login
  --username USERNAME  username to use for login
  --password PASSWORD  password to use for login
  --type TYPE          host/client
  --url URL            server REST api url
  --retry RETRY        retry limit
  --raw                output machine-readable json
  --explain            output verbose explanation of mapping of CLI calls to
                       HTTPS API endpoints
  --api-key API_KEY    api key. defaults to using the one stored in
                       ~/.vast_api_key

Creates a new account that is considered a child of your current account as defined via the API key.

﻿
create team
Create a new team

usage: vastai create-team --team_name TEAM_NAME

options:
  -h, --help            show this help message and exit
  --team_name TEAM_NAME
                        name of the team
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

﻿
create team-role
Add a new role to your

usage: vastai create team-role name --permissions PERMISSIONS

options:
  -h, --help            show this help message and exit
  --name NAME           name of the role
  --permissions PERMISSIONS
                        file path for json encoded permissions, look in the
                        docs for more information
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

﻿
delete api-key
Remove an api-key

usage: vastai delete api-key ID

positional arguments:
  ID                 id of apikey to remove

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
delete autoscaler
Delete an autoscaler group

usage: vastai delete autoscaler ID 

positional arguments:
  ID                 id of group to delete

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Note that deleteing an autoscaler group doesn't automatically destroy all the instances that are associated with your autoscaler group.
Example: vastai delete autoscaler 4242

﻿
delete overlay
Deletes an overlay

﻿

usage: vastai delete overlay OVERLAY_ID

positional arguments:
 overlay_id         ID of overlay to delete

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key

﻿
destroy instance
Destroy an instance (irreversible, deletes data)

usage: vastai destroy instance id [-h] [--api-key API_KEY] [--raw]

positional arguments:
  id                 id of instance to delete

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Perfoms the same action as pressing the "DESTROY" button on the website at https://console.vast.ai/instances/
Example: vastai destroy instance 4242

﻿
destroy instances
Destroy a list of instances (irreversible, deletes

usage: vastai destroy instances [--raw] <id>

positional arguments:
  ids                ids of instance to destroy

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
destroy team
Destroy your team

usage: vastai destroy team

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
execute
Execute a (constrained) remote command on a machine

usage: vastai execute ID COMMAND

positional arguments:
  ID                 id of instance to execute on
  COMMAND            bash command surrounded by single quotes

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

examples:
  vastai execute 99999 'ls -l -o -r'
  vastai execute 99999 'rm -r home/delete_this.txt'
  vastai execute 99999 'du -d2 -h'

available commands:
  ls                 List directory contents
  rm                 Remote files or directories
  du                 Summarize device usage for a set of files

Return value:
Returns the output of the command which was executed on the instance, if successful. May take a few seconds to retrieve the results.

﻿
generate pdf-invoices
usage: vastai generate pdf-invoices [OPTIONS]

options:
  -h, --help            show this help message and exit
  -q, --quiet           only display numeric ids
  -s START_DATE, --start_date START_DATE
                        start date and time for report. Many formats accepted
                        (optional)
  -e END_DATE, --end_date END_DATE
                        end date and time for report. Many formats accepted
                        (optional)
  -c, --only_charges    Show only charge items.
  -p, --only_credits    Show only credit items.
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

﻿
join overlay
Attaches an instance to an overlay network

usage: vastai join overlay OVERLAY_NAME INSTANCE_ID

positional arguments:
 name               Overlay network name to join instance to.
 instance_id        Instance ID to add to overlay.

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key
 --version          show version

Adds an instance to a compatible overlay network.

﻿
invite team-member
Invite a team member

usage: vastai invite team-member --email EMAIL --role ROLE

options:
  -h, --help         show this help message and exit
  --email EMAIL      email of user to be invited
  --role ROLE        role of user to be invited
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
label instance
Assign a string label to an instance

usage: vastai label instance <id> <label>

positional arguments:
  id                 id of instance to label
  label              label to set

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
logs
Get the logs for an instance

usage: vastai logs [OPTIONS] INSTANCE_ID

positional arguments:
  INSTANCE_ID        id of instance

options:
  -h, --help         show this help message and exit
  --tail TAIL        Number of lines to show from the end of the logs (default
                     '1000')
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
prepay instance
Deposit credits into reserved instance.

usage: vastai prepay instance <id> <amount>

positional arguments:
  id                 id of instance to prepay for
  amount             amount of instance credit prepayment (default discount
                     func of 0.2 for 1 month, 0.3 for 3 months)

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
reboot instance
Reboot (stop/start) an instance

usage: vastai reboot instance <id> [--raw]

positional arguments:
  id                 id of instance to reboot

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Instance is stopped and started without any risk of losing GPU priority.

﻿
remove team-member
Remove a team member

usage: vastai remove team-member ID

positional arguments:
  ID                 id of user to remove

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
remove team-role
Remove a role from your team

usage: vastai remove team-role NAME

positional arguments:
  NAME               name of the role

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
reports
Get the user reports for a given machine

usage: vastai reports m_id

positional arguments:
  m_id               machine id

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
reset api-key
Reset your api-key (get new key from website).

usage: vastai reset api-key

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
scp-url
scp url helper

usage: vastai scp-url ID

positional arguments:
  id                 id

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
search offers
Search for instance types using custom query

usage: vastai search offers [--help] [--api-key API_KEY] [--raw] <query>

positional arguments:
  query                 Query to search for. default: 'external=false
                        rentable=true verified=true', pass -n to ignore
                        default

options:
  -h, --help            show this help message and exit
  -t TYPE, --type TYPE  Show 'on-demand', 'reserved', or 'bid'(interruptible)
                        pricing. default: on-demand
  -i, --interruptible   Alias for --type=bid
  -b, --bid             Alias for --type=bid
  -r, --reserved        Alias for --type=reserved
  -d, --on-demand       Alias for --type=on-demand
  -n, --no-default      Disable default query
  --disable-bundling    Show identical offers. This request is more heavily
                        rate limited.
  --storage STORAGE     Amount of storage to use for pricing, in GiB.
                        default=5.0GiB
  -o ORDER, --order ORDER
                        Comma-separated list of fields to sort on. postfix
                        field with - to sort desc. ex: -o
                        'num_gpus,total_flops-'. default='score-'
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

Query syntax:

    query = comparison comparison...
    comparison = field op value
    field = <name of a field>
    op = one of: <, <=, ==, !=, >=, >, in, notin
    value = <bool, int, float, etc> | 'any'
    bool: True, False

note: to pass '>' and '<' on the command line, make sure to use quotes
note: to encode a string query value (ie for gpu_name), replace any spaces ' ' with underscore '_'

Examples:

    # search for somewhat reliable single RTX 3090 instances, filter out any duplicates or offers that conflict with our existing stopped instances
    vastai search offers 'reliability > 0.98 num_gpus=1 gpu_name=RTX_3090 rented=False'

    # search for datacenter gpus with minimal compute_cap and total_flops
    vastai search offers 'compute_cap > 610 total_flops > 5 datacenter=True'

    # search for reliable machines with at least 4 gpus, unverified, order by num_gpus, allow duplicates
    vastai search offers 'reliability > 0.99  num_gpus>=4 verified=False rented=any' -o 'num_gpus-'

Available fields:

      Name                  Type       Description

    bw_nvlink               float     bandwidth NVLink
    compute_cap:            int       cuda compute capability*100  (ie:  650 for 6.5, 700 for 7.0)
    cpu_cores:              int       # virtual cpus
    cpu_cores_effective:    float     # virtual cpus you get
    cpu_ram:                float     system RAM in gigabytes
    cuda_vers:              float     machine max supported cuda version (based on driver version)
    datacenter:             bool      show only datacenter offers
    direct_port_count       int       open ports on host's router
    disk_bw:                float     disk read bandwidth, in MB/s
    disk_space:             float     disk storage space, in GB
    dlperf:                 float     DL-perf score  (see FAQ for explanation)
    dlperf_usd:             float     DL-perf/$
    dph:                    float     $/hour rental cost
    driver_version          string    machine's nvidia driver version as 3 digit string ex. "535.86.05"
    duration:               float     max rental duration in days
    external:               bool      show external offers in addition to datacenter offers
    flops_usd:              float     TFLOPs/$
    geolocation:            string    Two letter country code. Works with operators =, !=, in, not in (e.g. geolocation not in [XV,XZ])
    gpu_mem_bw:             float     GPU memory bandwidth in GB/s
    gpu_name:               string    GPU model name (no quotes, replace spaces with underscores, ie: RTX_3090 rather than 'RTX 3090')
    gpu_ram:                float     GPU RAM in GB
    gpu_frac:               float     Ratio of GPUs in the offer to gpus in the system
    gpu_display_active:     bool      True if the GPU has a display attached
    has_avx:                bool      CPU supports AVX instruction set.
    id:                     int       instance unique ID
    inet_down:              float     internet download speed in Mb/s
    inet_down_cost:         float     internet download bandwidth cost in $/GB
    inet_up:                float     internet upload speed in Mb/s
    inet_up_cost:           float     internet upload bandwidth cost in $/GB
    machine_id              int       machine id of instance
    min_bid:                float     current minimum bid price in $/hr for interruptible
    num_gpus:               int       # of GPUs
    pci_gen:                float     PCIE generation
    pcie_bw:                float     PCIE bandwidth (CPU to GPU)
    reliability:            float     machine reliability score (see FAQ for explanation)
    rentable:               bool      is the instance currently rentable
    rented:                 bool      allow/disallow duplicates and potential conflicts with existing stopped instances
    storage_cost:           float     storage cost in $/GB/month
    static_ip:              bool      is the IP addr static/stable
    total_flops:            float     total TFLOPs from all GPUs
    ubuntu_version          string    host machine ubuntu OS version
    verified:               bool      is the machine verified

﻿
set api-key
Set api-key (get your api-key from the console/CLI)

usage: vastai set api-key APIKEY

positional arguments:
  new_api_key        Api key to set as currently logged in user

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show api-key
Show an api-key

usage: vastai show api-key

positional arguments:
  id                 id of apikey to get

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show api-keys
List your api-keys associated with your account

usage: vastai show api-keys

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show autoscalers
Display user's current autoscaler groups

usage: vastai show autoscalers [--api-key API_KEY]

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Example: vastai show autoscalers

﻿
show connections
Displays user's cloud connections

usage: vastai show connections [--api-key API_KEY] [--raw]

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show earnings
Get machine earning history reports

usage: vastai show earnings [OPTIONS]

options:
  -h, --help            show this help message and exit
  -q, --quiet           only display numeric ids
  -s START_DATE, --start_date START_DATE
                        start date and time for report. Many formats accepted
  -e END_DATE, --end_date END_DATE
                        end date and time for report. Many formats accepted
  -m MACHINE_ID, --machine_id MACHINE_ID
                        Machine id (optional)
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

﻿
show instance
Display user's current instances

usage: vastai show instance [--api-key API_KEY] [--raw]

positional arguments:
  id                 id of instance to get

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show instances
Display user's current instances

usage: vastai show instances [OPTIONS] [--api-key API_KEY] [--raw]

options:
  -h, --help         show this help message and exit
  -q, --quiet        only display numeric ids
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show invoices
Get billing history reports

usage: vastai show invoices [OPTIONS]

options:
  -h, --help            show this help message and exit
  -q, --quiet           only display numeric ids
  -s START_DATE, --start_date START_DATE
                        start date and time for report. Many formats accepted
                        (optional)
  -e END_DATE, --end_date END_DATE
                        end date and time for report. Many formats accepted
                        (optional)
  -c, --only_charges    Show only charge items.
  -p, --only_credits    Show only credit items.
  --instance_label INSTANCE_LABEL
                        Filter charges on a particular instance label (useful
                        for autoscaler groups)
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

﻿
show ipaddrs
Display user's history of ip addresses

usage: vastai show ipaddrs [--api-key API_KEY] [--raw]

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show overlays
Shows the client's created overlay networks

﻿

usage: vastai show overlays

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key
 --version          show version

Show overlays associated with your account.

﻿
show subaccounts
Get current subaccounts

usage: vastai show subaccounts [OPTIONS]

options:
  -h, --help         show this help message and exit
  -q, --quiet        display subaccounts from current user
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show team-members
Show your team members

usage: vastai show team-members

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show team-role
Show your team role

usage: vastai show team-role NAME

positional arguments:
  NAME               name of the role

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show team-roles
Show roles for a team

usage: vastai show team-roles

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
show user
Get current user data

usage: vastai show user [OPTIONS]

options:
  -h, --help         show this help message and exit
  -q, --quiet        display information about user
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Shows stats for logged-in user. These include user balance, email, and ssh key. Does not show API key.

﻿
ssh-url
ssh url helper

usage: vastai ssh-url ID

positional arguments:
  id                 id of instance

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
start instance
Start a stopped instance

usage: vastai start instance <id> [--raw]

positional arguments:
  id                 id of instance to start/restart

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

This command attempts to bring an instance from the "stopped" state into the "running" state. This is subject to resource availability on the machine that the instance is located on.
If your instance is stuck in the "scheduling" state for more than 30 seconds after running this, it likely means that the required resources on the machine to run your instance are currently unavailable.
Examples:
    vastai start instances $(vastai show instances -q)
    vastai start instance 329838

﻿
start instances
Start a list of instances

usage: vastai start instances [--raw] ID0 ID1 ID2...

positional arguments:
  ids                ids of instance to start

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
stop instance
Stop a running instance

usage: vastai stop instance [--raw] ID

positional arguments:
  id                 id of instance to stop

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

This command brings an instance from the "running" state into the "stopped" state. When an instance is "stopped" all of your data on the instance is preserved,
and you can resume use of your instance by starting it again. Once stopped, starting an instance is subject to resource availability on the machine that the instance is located on.
There are ways to move data off of a stopped instance, which are described here: https://docs.vast.ai/instances/data-movement

﻿
stop instances
Stop a list of instances

usage: vastai stop instances [--raw] ID0 ID1 ID2...

positional arguments:
  ids                ids of instance to stop

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Examples:
    vastai stop instances $(vastai show instances -q)
    vastai stop instances 329838 984849

﻿
transfer credit
Transfer credits to another account

usage: vastai transfer credit RECIPIENT AMOUNT

positional arguments:
  recipient          email of recipient account
  amount             $dollars of credit to transfer

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Transfer (amount) credits to account with email (recipient).

﻿
update autoscaler
Update an existing autoscale group

usage: vastai update autoscaler ID [OPTIONS]

positional arguments:
  ID                    id of autoscale group to update

options:
  -h, --help            show this help message and exit
  --min_load MIN_LOAD   minimum floor load in perf units/s (token/s for LLms)
  --target_util TARGET_UTIL
                        target capacity utilization (fraction, max 1.0,
                        default 0.9)
  --cold_mult COLD_MULT
                        cold/stopped instance capacity target as multiple of
                        hot capacity target (default 2.5)
  --gpu_ram GPU_RAM     estimated GPU RAM req (independent of search string)
  --template_hash TEMPLATE_HASH
                        template hash
  --template_id TEMPLATE_ID
                        template id
  --search_params SEARCH_PARAMS
                        search param string for search offers ex: "gpu_ram>=23
                        num_gpus=2 gpu_name=RTX_4090 inet_down>200
                        direct_port_count>2 disk_space>=64"
  --launch_args LAUNCH_ARGS
                        launch args string for create instance ex: "--onstart
                        onstart_wget.sh --env '-e ONSTART_PATH=https://s3.amaz
                        onaws.com/vast.ai/onstart_OOBA.sh' --image
                        atinoda/text-generation-webui:default-nightly --disk
                        64"
  --endpoint_name ENDPOINT_NAME
                        deployment endpoint name (allows multiple autoscale
                        groups to share same deployment endpoint)
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

Example: vastai update autoscaler 4242 --min_load 100 --target_util 0.9 --cold_mult 2.0 --search_params "gpu_ram>=23 num_gpus=2 gpu_name=RTX_4090 inet_down>200 direct_port_count>2 disk_space>=64" --launch_args "--onstart onstart_wget.sh  --env '-e ONSTART_PATH=https://s3.amazonaws.com/vast.ai/onstart_OOBA.sh' --image atinoda/text-generation-webui:default-nightly --disk 64" --gpu_ram 32.0 --endpoint_name "LLama"

﻿
update team-role
Update an existing team role

usage: vastai update team-role ID --name NAME --permissions PERMISSIONS

positional arguments:
  ID                    id of the role

options:
  -h, --help            show this help message and exit
  --name NAME           name of the template
  --permissions PERMISSIONS
                        file path for json encoded permissions, look in the
                        docs for more information
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

﻿
Host Commands
create cluster
Registers a new locally-networked cluster with the Vast. 

﻿

usage: vastai create cluster SUBNET MANAGER_ID

positional arguments:
 subnet             local subnet for cluster, ex: '0.0.0.0/24'
 manager_id         Machine ID of manager node in cluster. Must exist already.

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key
 --version          show version

Create Vast Cluster by defining a local subnet and manager id.

﻿
delete cluster
Deregisters a cluster

usage: vastai delete cluster CLUSTER_ID

positional arguments:
 cluster_id         ID of cluster to delete

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key
 --version          show version

Delete Vast Cluster

﻿
join cluster
Registers a machine or list of machines as a member of a cluster. 

usage: vastai join cluster CLUSTER_ID MACHINE_IDS

positional arguments:
 cluster_id         ID of cluster to add machine to
 machine_ids        machine id(s) to join cluster

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key
 --version          show version

Join's Machine to Vast Cluster

﻿
list machine
[Host] list a machine for rent

usage: vastai list machine id [--price_gpu PRICE_GPU] [--price_inetu PRICE_INETU] [--price_inetd PRICE_INETD] [--api-key API_KEY]

positional arguments:
  id                    id of machine to list

options:
  -h, --help            show this help message and exit
  -g PRICE_GPU, --price_gpu PRICE_GPU
                        per gpu rental price in $/hour (price for active
                        instances)
  -s PRICE_DISK, --price_disk PRICE_DISK
                        storage price in $/GB/month (price for inactive
                        instances), default: $0.15/GB/month
  -u PRICE_INETU, --price_inetu PRICE_INETU
                        price for internet upload bandwidth in $/GB
  -d PRICE_INETD, --price_inetd PRICE_INETD
                        price for internet download bandwidth in $/GB
  -r DISCOUNT_RATE, --discount_rate DISCOUNT_RATE
                        Max long term prepay discount rate fraction, default:
                        0.4
  -m MIN_CHUNK, --min_chunk MIN_CHUNK
                        minimum amount of gpus
  -e END_DATE, --end_date END_DATE
                        unix timestamp of the available until date (optional)
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

Performs the same action as pressing the "LIST" button on the site https://cloud.vast.ai/host/machines.
On the end date the listing will expire and your machine will unlist. However any existing client jobs will still remain until ended by their owners.
Once you list your machine and it is rented, it is extremely important that you don't interfere with the machine in any way.
If your machine has an active client job and then goes offline, crashes, or has performance problems, this could permanently lower your reliability rating.
We strongly recommend you test the machine first and only list when ready.

﻿
remove-machine-from-cluster
Deregisters a machine from a cluster, changing the manager node if the machine removed is the only manager. 

﻿

usage: vastai remove-machine-from-cluster CLUSTER_ID MACHINE_ID NEW_MANAGER_ID

positional arguments:
 cluster_id         ID of cluster you want to remove machine from.
 machine_id         ID of machine to remove from cluster.
 new_manager_id     ID of machine to promote to manager. Must already be in cluster

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key
 --version          show version

Removes machine from cluster and also reassigns manager ID,
if we're removing the manager node

﻿
remove defjob
[Host] Delete default jobs

usage: vastai remove defjob id

positional arguments:
  id                 id of machine to remove default instance from

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
schedule maint
[Host] Schedule upcoming maint window

usage: vastai schedule maintenance id [--sdate START_DATE --duration DURATION]

positional arguments:
  id                   id of machine to schedule maintenance for

options:
  -h, --help           show this help message and exit
  --sdate SDATE        maintenance start date in unix epoch time (UTC seconds)
  --duration DURATION  maintenance duration in hours
  --url URL            server REST api url
  --retry RETRY        retry limit
  --raw                output machine-readable json
  --explain            output verbose explanation of mapping of CLI calls to
                       HTTPS API endpoints
  --api-key API_KEY    api key. defaults to using the one stored in
                       ~/.vast_api_key

The proper way to perform maintenance on your machine is to wait until all active contracts have expired or the machine is vacant.
For unplanned or unscheduled maintenance, use this schedule maint command. That will notify the client that you have to take the machine down and that they should save their work.
You can specify a date and duration.
Example: vastai schedule maint 8207 --sdate 1677562671 --duration 0.5

﻿
set defjob
[Host] Create default jobs for a machine

usage: vastai set defjob id [--api-key API_KEY] [--price_gpu PRICE_GPU] [--price_inetu PRICE_INETU] [--price_inetd PRICE_INETD] [--image IMAGE] [--args ...]

positional arguments:
  id                    id of machine to launch default instance on

options:
  -h, --help            show this help message and exit
  --price_gpu PRICE_GPU
                        per gpu rental price in $/hour
  --price_inetu PRICE_INETU
                        price for internet upload bandwidth in $/GB
  --price_inetd PRICE_INETD
                        price for internet download bandwidth in $/GB
  --image IMAGE         docker container image to launch
  --args ...            list of arguments passed to container launch
  --url URL             server REST api url
  --retry RETRY         retry limit
  --raw                 output machine-readable json
  --explain             output verbose explanation of mapping of CLI calls to
                        HTTPS API endpoints
  --api-key API_KEY     api key. defaults to using the one stored in
                        ~/.vast_api_key

Performs the same action as creating a background job at https://cloud.vast.ai/host/create.

﻿
set min-bid
[Host] Set the minimum bid/rental price for a machine

usage: vastai set min_bid id [--price PRICE]

positional arguments:
  id                 id of machine to set min bid price for

options:
  -h, --help         show this help message and exit
  --price PRICE      per gpu min bid price in $/hour
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

Change the current min bid price of machine id to PRICE.

﻿
show clusters
Shows information about the host's clusters

usage: vastai show clusters

options:
 -h, --help         show this help message and exit
 --url URL          server REST api url
 --retry RETRY      retry limit
 --raw              output machine-readable json
 --explain          output verbose explanation of mapping of CLI calls to HTTPS API endpoints
 --curl             show a curl equivalency to the call
 --api-key API_KEY  api key. defaults to using the one stored in /home/edgarlin/.config/vastai/vast_api_key
 --version          show version

Show clusters associated with your account.

﻿
show machines
[Host] Show hosted machines

usage: vastai show machines [OPTIONS]

options:
  -h, --help         show this help message and exit
  -q, --quiet        only display numeric ids
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
unlist machine
[Host] Unlist a listed machine

usage: vastai unlist machine <id>

positional arguments:
  id                 id of machine to unlist

options:
  -h, --help         show this help message and exit
  --url URL          server REST api url
  --retry RETRY      retry limit
  --raw              output machine-readable json
  --explain          output verbose explanation of mapping of CLI calls to
                     HTTPS API endpoints
  --api-key API_KEY  api key. defaults to using the one stored in
                     ~/.vast_api_key

﻿
﻿

Updated 12 Jul 2025
Did this page help you?

Yes

No
PREVIOUS
Overview & quickstart
NEXT
Permissions-and-authorization
TABLE OF CONTENTS
CLI Commands
Client Commands
cancel copy
cancel sync
change bid
cloud copy
copy
create api-key
create autoscaler
create instance
create overlay
create subaccount
create team
create team-role
delete api-key
delete autoscaler
delete overlay
destroy instance
destroy instances
destroy team
execute
generate pdf-invoices
join overlay
invite team-member
label instance
logs
prepay instance
reboot instance
remove team-member
remove team-role
reports
reset api-key
scp-url
search offers
set api-key
show api-key
show api-keys
show autoscalers
show connections
show earnings
show instance
show instances
show invoices
show ipaddrs
show overlays
show subaccounts
show team-members
show team-role
show team-roles
show user
ssh-url
start instance
start instances
stop instance
stop instances
transfer credit
update autoscaler
update team-role
Host Commands
create cluster
delete cluster
join cluster
list machine
remove-machine-from-cluster
remove defjob
schedule maint
set defjob
set min-bid
show clusters
show machines
unlist machine
search offers - API
