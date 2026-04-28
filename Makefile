.PHONY: help up down build restart logs test clean sockets

COMPOSE = docker compose

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

sockets: ## Create sockets directory if it doesn't exist
	@mkdir -p sockets

build: sockets ## Build all service images
	$(COMPOSE) build

up: sockets ## Start all services
	$(COMPOSE) up --build

down: ## Stop and remove all containers
	$(COMPOSE) down

stop: ## Stop containers without removing them
	$(COMPOSE) stop

restart: down up ## Restart all services

logs: ## Follow logs from all services
	$(COMPOSE) logs -f

logs-api1: ## Follow logs from api1
	$(COMPOSE) logs -f api1

logs-api2: ## Follow logs from api2
	$(COMPOSE) logs -f api2

logs-nginx: ## Follow logs from nginx
	$(COMPOSE) logs -f nginx

ps: ## Show running containers
	$(COMPOSE) ps

test: ## Run k6 load test
	k6 run test.js

clean: down ## Stop containers and clean sockets
	rm -rf sockets/*

clean-all: down ## Stop containers, clean sockets and remove images
	rm -rf sockets/*
	$(COMPOSE) down -v
