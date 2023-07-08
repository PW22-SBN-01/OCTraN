import logging, colorlog

################################################################
# COLOR LOGGING SETUP
################################################################
log = logging.getLogger()
log.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
	"%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
	datefmt=None,
	reset=True,
	log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},
	secondary_log_colors={},
	style='%'
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)