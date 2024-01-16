$prefix = "grc"
$location = "eastus"
$created_by = "gabriel_r"
$resource_group = $prefix + "-ml-rg"

$ml_ws_name = $prefix + "-ml-ws"
$ml_ws_description = "Ml Workspace from CLI."

$cur_date = Get-Date -Format "yyyy_MM_dd"

# Create Resource Group
az group create --name $resource_group
                --location $location
                --subscription $Env:azure_subscription
                --tags "created_by=$created_by" "created_at=$cur_date"

Start-Sleep -Seconds 20

# Create Azure Ml Ws
az ml workspace create --name $ml_ws_name
                       --location $location
                       --subscription $Env:azure_subscription
                       --resource-group $resource_group
                       --description $ml_ws_description
                       --tags "created_by=$created_by" "created_at=$cur_date"