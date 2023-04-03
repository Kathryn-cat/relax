# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relax
from tvm.contrib.cutlass.build import is_shape_valid_for_cutlass_matmul
from tvm.contrib.pickle_memoize import memoize
from tvm.relax.backend import get_patterns_with_prefix
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.script import relax as R
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder
