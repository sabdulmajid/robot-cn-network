#!/bin/bash
# Setup script for Robot CN Network development environment
# This script creates a virtual environment, installs dependencies, and runs tests

# Set error handling
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up development environment for Robot CN Network...${NC}"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install the package in development mode
echo -e "${YELLOW}Installing project in development mode...${NC}"
pip install -e ".[dev]"

# Run the visualization test
echo -e "${YELLOW}Running visualization tests to generate example outputs...${NC}"
python tests/test_visualization.py

# Copy visualization outputs to docs/images
echo -e "${YELLOW}Copying visualization outputs to docs/images...${NC}"
mkdir -p docs/images
cp outputs/example_visualizations/*.png docs/images/

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}You can now run the following commands:${NC}"
echo -e "  ${YELLOW}source venv/bin/activate${NC} - Activate the virtual environment"
echo -e "  ${YELLOW}robot-analyze --help${NC} - Get help on the analysis tool"
echo -e "  ${YELLOW}python tests/test_visualization.py${NC} - Generate example visualizations"
echo -e "\nVisit the docs directory to see documentation and example visualizations."
