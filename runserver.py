#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import argparse
from upload.app import app

parser = argparse.ArgumentParser(description="Upload")
parser.add_argument(
    "--port", "-p",
    type=int,
    help="Port to listen on",
    default=5050,
)
parser.add_argument(
    "--host",
    type=str,
    help="host",
    default='0.0.0.0',
)
args = parser.parse_args()

if __name__ == '__main__':
    flask_options = dict(
        host=args.host,
        debug=False,
        port=args.port,
        threaded=True,
    )

    app.run(**flask_options)
