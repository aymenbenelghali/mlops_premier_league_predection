param(
    [string]$Tag = "premier_league_prediction:latest"
)
Write-Host "Building image $Tag"
$env:DOCKER_BUILDKIT = "1"
docker build -t $Tag .
Write-Host "Built image: $Tag"
