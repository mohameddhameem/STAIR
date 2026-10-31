#!/bin/bash
# MedAgentBench's FHIR records server without Docker: the server, Java 17 and patient database unpacked from the authors'
# image (tools/pull_image.py jyxsu6/medagentbench:latest ~/stair-data/medagentbench-image).
#   source jobs/fhir_server.sh; fhir_start; ... use $FHIR_URL ...; fhir_stop
# Every job works on its own copy of the database (the tasks place orders, which write to it), on the node's local disk
# when there is room (else in the home folder), removed again by fhir_stop.
FHIR_IMAGE="${FHIR_IMAGE:-$HOME/stair-data/medagentbench-image}"

fhir_start() {
  local R="$FHIR_IMAGE/rootfs"
  local JAVA="$R/usr/lib/jvm/java-17-openjdk-amd64/bin/java"
  [ -x "$JAVA" ] || { echo "no Java in $R: run tools/pull_image.py first"; return 1; }
  local base="${TMPDIR:-/tmp}"
  local need=$(( $(stat -c %s "$R/data/test_db.mv.db") / 1024 + 1048576 ))
  if [ "$(df -Pk "$base" | awk 'NR==2 {print $4}')" -lt "$need" ]; then base="$HOME/stair-data/tmp"; fi
  FHIR_DB_DIR="$base/fhir-db-${SLURM_JOB_ID:-$$}"
  mkdir -p "$FHIR_DB_DIR"
  echo "copying the patient database to $FHIR_DB_DIR ..."
  cp "$R/data/test_db.mv.db" "$FHIR_DB_DIR/"
  FHIR_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
  FHIR_URL="http://localhost:$FHIR_PORT/fhir/"
  FHIR_LOG="${FHIR_LOG:-logs/fhir-${SLURM_JOB_ID:-$$}.log}"
  ( cd "$R/app" && exec "$JAVA" -Xmx${FHIR_HEAP:-12g} --class-path "$R/app/main.war" \
      "-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/,$R/app/extra-classes" \
      org.springframework.boot.loader.PropertiesLauncher \
      "--spring.config.location=file://$R/configs/application.yaml" \
      "--server.port=$FHIR_PORT" \
      "--spring.datasource.url=jdbc:h2:$FHIR_DB_DIR/test_db;DB_CLOSE_DELAY=-1" ) > "$FHIR_LOG" 2>&1 &
  FHIR_PID=$!
  echo "FHIR server starting (pid $FHIR_PID, port $FHIR_PORT, log $FHIR_LOG)"
  for i in $(seq 1 180); do                         # up to 15 minutes
    if curl -sf "${FHIR_URL}metadata" -o /dev/null; then echo "FHIR server ready at $FHIR_URL after $((i * 5)) s"; export FHIR_URL FHIR_PORT; return 0; fi
    if ! kill -0 "$FHIR_PID" 2>/dev/null; then echo "FHIR server stopped while starting; last lines of $FHIR_LOG:"; tail -20 "$FHIR_LOG"; return 1; fi
    sleep 5
  done
  echo "FHIR server not ready after 15 minutes; last lines of $FHIR_LOG:"; tail -20 "$FHIR_LOG"; fhir_stop; return 1
}

fhir_stop() {
  [ -n "$FHIR_PID" ] && kill "$FHIR_PID" 2>/dev/null && wait "$FHIR_PID" 2>/dev/null
  [ -n "$FHIR_DB_DIR" ] && rm -rf "$FHIR_DB_DIR"
}
