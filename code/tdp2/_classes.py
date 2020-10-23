import requests
import warnings
import sys


class AuthMixin(object):
    def __init__(self,
                 access_token=None, access_secret=None,
                 consumer_key=None, consumer_secret=None,
                 bearer_token=None, oauth_version=1,
                 auth_type='app'):
        self.access_token = access_token
        self.access_secret = access_secret
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.oauth_version = 1

        # Bearer token needs to be initialized after oauth_version.
        self.bearer_token = bearer_token

        # Set the authorization type, which in turn sets the rate limit.
        self.auth_type = auth_type

    @property
    def bearer_token(self):
        return self.__bearer_token
    @bearer_token.setter
    def bearer_token(self, value):
        if value is not None:
            self.oauth_version = 2
        self.__bearer_token = value

    @property
    def rate_limit(self):
        # Rate limit by Twitter developer authorization types: app or user
        #
        # app auth_type: 300 requests per 15-minute window
        # user auth_type: 900 requests per 15-minute window
        if self.auth_type=='app':
            return 9
        elif self.auth_type=='user':
            return 3
        else:
            raise ValueError('Invalid authorization type (must be \'app\' \
                              or \'user\'')


class Hydrator(AuthMixin):
    '''
    Tweet hydrator using the Twitter API v2.
    Currently supports hydration from a list of tweet IDs from the /2/tweets
    endpoint.

    Parameters
    ----------
    access_token : str, optional
        Twiter developer access token.
    access_secret : str, optional
        Twitter developer access secret.
    bearer_token : str, optional
        Twitter developer bearer token

    Methods
    -------
    hydrate_list(ids=[], args={})
        Hydrate tweets from a list of IDs.
        Only 100 tweets can be hydrated

    Examples
    --------
    >>> BEARER_TOKEN = os.environ['TWITTER_BEARER_TOKEN']
    >>> h = Hydrator(url='abc', bearer_token=BEARER_TOKEN)

    '''
    def __init__(self, **kwargs):
        super(Hydrator, self).__init__(**kwargs)
        # Set up variables for the GET request.
        self.endpoint_url = 'https://api.twitter.com/2/tweets'

    @property
    def header(self):
        '''
        Define the required authorization header for the GET query, using
        the provided bearer token.
        '''
        return {'Authorization': 'Bearer {}'.format(self.bearer_token)}

    @staticmethod
    def _warn(flag, n):
        if not flag:
            return
        if n > 100:
            msg="Hydrating large batches of tweets may take ;a very long time."
            warnings.warn(msg)

    @staticmethod
    def _parameterize(params, args):
        for k, v in args.items():
            params[k] =  ','.join(v)


    def hydrate_list(self, ids=[], args={}, warn=True):
        # Warn user that hydrating a large number of tweets
        # may take a very long time.
        self._warn(warn,len(ids))

        # Set up variables for the GET request.
        parameters = {}
        self._parameterize(parameters, args)

        # Query endpoint with a max of 100 tweet IDs at a time.
        payloads = []
        for i in range(0, len(ids), 100):
            j = min(i + 100, len(ids))  # For end of list.

            self._parameterize(parameters, {'ids':ids[i:j]})
            payload = requests.get(self.endpoint_url, params=parameters,
                                   headers=self.header)
            payloads.append(payload)

        # Return list of 100-tweet-batch payloads. If less than 100 tweet IDs
        # were provided then simply return the json payload.
        if len(payloads) == 1:
            return payloads[0]
        else:
            return payloads
