# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from loggers import Timer

logger = logging.getLogger(__name__)

class Callback:
    def __init__(self, name = None, cond = None, initializer = None, ** _):
        self.name   = name or self.__class__.__name__
        self.cond   = cond
        self.initializer    = initializer
        
        self.built  = False
    
    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)
    
    def build(self):
        self.built = True
    
    def __call__(self, infos, output, ** kwargs):
        if self.cond is not None and not self.cond(** output):
            return
        elif self.initializer:
            for k, fn in self.initializer.items():
                if k not in output: output[k] = fn(** output)
        
        if not self.built: self.build()
            
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('- Apply {}'.format(self))
        
        self.apply(infos = infos, output = output, ** kwargs)
    
    def apply(self, infos, output, ** kwargs):
        raise NotImplementedError()

    def join(self):
        pass
