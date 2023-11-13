#!/bin/bash
set -e

# Aggiorna PUID e PGID se sono passati come variabili d'ambiente
if [ ! -z "$PUID" ]; then
    usermod -u $PUID server
fi

if [ ! -z "$PGID" ]; then
    groupmod -g $PGID server
fi

# Cambia l'ownership delle directory se PUID o PGID sono cambiati
chown -R server:server /home/server

# Esegui il comando passato al container (es. python server.py)
exec "$@"