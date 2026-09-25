#!/usr/bin/env bash
# Start / stop / restart the qw server on this box.
#   server/ctl.sh start [extra python -m qwserve args]   |   stop   |   restart ...   |   status
set -u
DIR=$(cd "$(dirname "$0")/.." && pwd)
PY=${QW_PYTHON:-$HOME/qwenv/bin/python}
LOG=${QW_LOG:-/tmp/qw_server.log}
PIDFILE=${QW_PIDFILE:-/tmp/qw_server.pid}
PORT=${QW_PORT:-8000}

stop() {
  [ -f $PIDFILE ] && kill $(cat $PIDFILE) 2>/dev/null
  for i in $(seq 1 30); do [ -f $PIDFILE ] && kill -0 $(cat $PIDFILE) 2>/dev/null || break; sleep 1; done
  rm -f $PIDFILE
}
start() {
  cd $DIR
  PYTHONPATH=$DIR/server nohup $PY -m qwserve --port $PORT "$@" > $LOG 2>&1 &
  echo $! > $PIDFILE
  for i in $(seq 1 120); do
    curl -s -m2 localhost:$PORT/health >/dev/null && { tail -1 $LOG; echo "up (pid $(cat $PIDFILE))"; return 0; }
    kill -0 $(cat $PIDFILE) 2>/dev/null || { echo "server died:"; tail -20 $LOG; return 1; }
    sleep 5
  done
  echo "timed out"; return 1
}
case ${1:-} in
  start) shift; start "$@" ;;
  stop) stop ;;
  restart) shift; stop; start "$@" ;;
  status) curl -s -m2 localhost:$PORT/health && echo || echo down ;;
  *) echo "usage: $0 start|stop|restart|status [args]"; exit 2 ;;
esac
