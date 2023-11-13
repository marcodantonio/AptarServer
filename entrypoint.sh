#!/bin/bash
set -e

# Aggiorna PUID e PGID se sono passati come variabili d'ambiente e sono diversi dai valori correnti
if [ ! -z "$PUID" ] && [ "$PUID" != "$(id -u server)" ]; then
    usermod -u $PUID server
fi

if [ ! -z "$PGID" ] && [ "$PGID" != "$(id -g server)" ]; then
    groupmod -g $PGID server
fi

# Cambia l'ownership delle directory se PUID o PGID sono cambiati
chown -R server:server /home/server

# Esegui il comando passato al container come utente 'server'
exec gosu server "$@"