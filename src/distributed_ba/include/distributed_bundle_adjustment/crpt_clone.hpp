#pragma once

// Note that this file is a 1-to-1 copy of the aslam version of it. 

namespace dba {

// Implements the cloneable concept using the CRTP and constructor inheritance.
// Example:
//  class Vehicle {
//    Vehicle() = delete;
//    Vehicle(int speed) : speed_(speed) {};
//    virtual Vehicle* clone() const = 0;  // Make sure to add the pure virtual function clone
//    const int speed_;
//  };
//
//  class Car : public aslam::Cloneable<Vehicle, Car> {
//    Car(int speed) : Base(speed) {};     // Use the "Base" typedef defined by the Cloneable class
//  };
template<typename BaseType, typename DerivedType>
class Cloneable : public BaseType {
 public:
  typedef Cloneable Base;
  using BaseType::BaseType;

  /// Method to clone this instance
  virtual auto clone() const ->BaseType* {
    return new DerivedType(static_cast<const DerivedType&>(*this));
  };

  virtual ~Cloneable() {};
};


}  // namespace dba
