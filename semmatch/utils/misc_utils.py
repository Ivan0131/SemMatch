# coding=utf-8

import re


def snake_to_camel(name):
    s1 = re.sub('_([a-zA-Z0-9])', lambda m: m.group(1).upper(), name)
    return s1[0].upper()+s1[1:]


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
