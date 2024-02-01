
import copy

def iter_properties_of_class(cls):
    for varname in vars(cls):
        value = getattr(cls, varname)
        if isinstance(value, property):
            yield varname

def properties(inst):
    result = {}
    for cls in inst.__class__.mro():
        abandon_properties = getattr(cls, '__abandon_properties__', [])
        for varname in iter_properties_of_class(cls):
            if varname[0] == "_":
                continue
            if varname in abandon_properties:
                continue
            try:
                tmp = getattr(inst, varname)
            except (AttributeError, RuntimeError, KeyboardInterrupt):
                continue
            if varname == "positions":
                tmp = list(tmp.keys())
            if hasattr(tmp, '__simple_object__'):
                result[varname] = tmp.__simple_object__()
            else:
                result[varname] = tmp
    return result

def property_repr(inst):
    return "%s(%s)" % (inst.__class__.__name__, properties(inst))

class dict2class:
    # __repr__  = property_repr

    def __init__(self, dic):
        self.__dict__ = copy.copy(dic)

    @property
    def algorithm(self):
        return self.__dict__["algorithm"]

    @property
    def order_book_id(self):
        return self.__dict__["order_book_id"]

    @property
    def symbol(self):
        return self.__dict__["symbol"]



class dict2obj(object):
    def __init__(self, dict_data):
        for key in dict_data:
            if isinstance(dict_data[key], dict):
                if key != "config_record" and key != "data_type":
                    setattr(self, key, dict2obj(dict_data[key]))
                else:
                    setattr(self, key, dict_data[key])
            else:
                setattr(self, key, dict_data[key])
    def __repr__(self):
        return "%s" %self.__dict__

def obj2dict(obj_data):
    dict_data = {}
    for property_name in dir(obj_data):
        if "__" not in property_name:
            dict_data.update({property_name: getattr(obj_data, property_name)})
    return dict_data
