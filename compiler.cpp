#include "MiniAPLJIT.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

class ASTNode;

// -------------------------------------------------
// Miscellaneous helper functions
// -------------------------------------------------

// Log an error message to standard error.
void LogError(std::string_view S) { llvm::errs() << S; }

bool is_int(std::string_view str) {
  // Check with regex (does not accept leading zeroes before first digit)
  static constexpr int max_digits = std::numeric_limits<int>::digits10;
  static const std::string ub = std::to_string(max_digits - 1);
  static const std::regex int_re("^\\s*([+-]?[1-9]\\d{0," + ub + "}|0)\\s*$");

  return std::regex_match(std::string(str), int_re);
}

// -------------------------------------------------
// Type information for MiniAPL programs
// -------------------------------------------------

enum ExprType { EXPR_TYPE_SCALAR, EXPR_TYPE_FUNCALL, EXPR_TYPE_VARIABLE };

class MiniAPLArrayType {
public:
  std::vector<int> dimensions;

  int Cardinality() {
    int C = 1;
    for (int D : dimensions) {
      C *= D;
    }
    return C;
  }

  int length(const int dim) { return dimensions.at(dim); }

  int dimension() { return dimensions.size(); }
};

std::ostream &operator<<(std::ostream &out, MiniAPLArrayType &tp) {
  out << "[";
  int i = 0;
  for (int T : tp.dimensions) {
    out << T;
    if (i + 1 < tp.dimensions.size()) {
      out << ", ";
    }
    i++;
  }
  out << "]";
  return out;
}

// -------------------------------------------------
// AST classes
// -------------------------------------------------

// The base class for all expression nodes.
class ASTNode {
public:
  virtual ~ASTNode() = default;

  virtual Value *codegen(Function *F) = 0;
  virtual ExprType GetType() = 0;
  virtual void Print(std::ostream &out) {}
};

std::ostream &operator<<(std::ostream &out, ASTNode &tp) {
  tp.Print(out);
  return out;
}

class StmtAST : public ASTNode {
public:
  virtual bool IsAssign() = 0;
};

class ProgramAST : public ASTNode {
public:
  std::vector<std::unique_ptr<StmtAST>> Stmts;
  Value *codegen(Function *F) override;
  virtual ExprType GetType() override { return EXPR_TYPE_FUNCALL; }
};

class ExprStmtAST : public StmtAST {
public:
  std::unique_ptr<ASTNode> Val;

  bool IsAssign() override { return false; }
  ExprStmtAST(std::unique_ptr<ASTNode> Val_) : Val(std::move(Val_)) {}
  Value *codegen(Function *F) override;
  virtual ExprType GetType() override { return EXPR_TYPE_FUNCALL; }

  virtual void Print(std::ostream &out) override { Val->Print(out); }
};

class VariableASTNode : public ASTNode {

public:
  std::string Name;
  VariableASTNode(std::string_view Name) : Name(Name) {}

  Value *codegen(Function *F) override;

  virtual ExprType GetType() override { return EXPR_TYPE_VARIABLE; }

  virtual void Print(std::ostream &out) override { out << Name; }
};

class AssignStmtAST : public StmtAST {
public:
  std::unique_ptr<VariableASTNode> Name;
  std::unique_ptr<ASTNode> RHS;

  bool IsAssign() override { return true; }
  Value *codegen(Function *F) override;

  std::string GetName() const { return Name->Name; }

  AssignStmtAST(std::string_view Name_, std::unique_ptr<ASTNode> val_)
      : Name(new VariableASTNode(Name_)), RHS(std::move(val_)) {}
  virtual ExprType GetType() override { return EXPR_TYPE_FUNCALL; }
  virtual void Print(std::ostream &out) override {
    out << "assign ";
    Name->Print(out);
    out << " = ";
    RHS->Print(out);
  }
};

class NumberASTNode : public ASTNode {
public:
  int Val;
  NumberASTNode(int Val) : Val(Val) {}

  Value *codegen(Function *F) override;

  virtual ExprType GetType() override { return EXPR_TYPE_SCALAR; }

  virtual void Print(std::ostream &out) override { out << Val; }
};

class CallASTNode : public ASTNode {

public:
  std::string Callee;
  std::vector<std::unique_ptr<ASTNode>> Args;
  CallASTNode(std::string_view Callee,
              std::vector<std::unique_ptr<ASTNode>> Args)
      : Callee(Callee), Args(std::move(Args)) {}

  Value *codegen(Function *F) override;
  virtual ExprType GetType() override { return EXPR_TYPE_FUNCALL; }
  virtual void Print(std::ostream &out) override {
    out << Callee << "(";
    for (int i = 0; i < Args.size(); i++) {
      Args.at(i)->Print(out);
      if (i + 1 < Args.size()) {
        out << ", ";
      }
    }
    out << ")";
  }
};

// ---------------------------------------------------------------------------
// Some global variables used in parsing, type-checking, and code generation.
// ---------------------------------------------------------------------------

// NOTE: These tables will be helpful for "codegen" methods.
static std::map<ASTNode *, MiniAPLArrayType> TypeTable;
static std::map<std::string, Value *> ValueTable;
static std::unique_ptr<LLVMContext> TheContext =
    std::make_unique<LLVMContext>();
// NOTE: You will probably want to use the Builder in the "codegen" methods
static IRBuilder<> Builder(*TheContext);
static std::unique_ptr<Module> TheModule;
static std::map<std::string, Value *> NamedValues;

// ---------------------------------------------------------------------------
// LLVM codegen helpers
// ---------------------------------------------------------------------------

// Returns an integer type with the given width.
IntegerType *intTy(const int width) {
  return IntegerType::get(*TheContext, width);
}

// Returns a constant integer with the given width and value.
ConstantInt *intConst(const int width, const int i) {
  ConstantInt *const_int32 = ConstantInt::get(
      *TheContext, APInt(width, StringRef(std::to_string(i)), 10));
  return const_int32;
}

// Helper function to get or initialize the C++ `printf` function.
static Function *__GetOrCreatePrintf(Module *M) {
  Function *func_printf;
  if (func_printf = M->getFunction("printf"); func_printf)
    return func_printf;

  LLVMContext &Ctx = M->getContext();
  FunctionType *FuncTy = FunctionType::get(
      IntegerType::get(Ctx, 32),
      llvm::PointerType::get(llvm::IntegerType::get(Ctx, 8), 0), true);

  func_printf =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, "printf", M);
  func_printf->setCallingConv(CallingConv::C);
  return func_printf;
}

// NOTE: This utility function generates LLVM IR to print out the std::string
// `to_print`, e.g., CreatePrintfStr(M, BB, "XXX") will print "XXX" when
// executed.
void CreatePrintfStr(Module *mod, BasicBlock *bb, std::string_view to_print) {
  Function *func_printf = __GetOrCreatePrintf(mod);

  IRBuilder<> builder(*TheContext);
  builder.SetInsertPoint(bb);
  Value *S = builder.CreateGlobalStringPtr(to_print);
  assert(S && "invalid string");
  std::vector<Value *> int32_call_params;
  int32_call_params.push_back(S);

  CallInst::Create(func_printf, int32_call_params, "call", bb);
}

// NOTE: This utility function generates code that prints out the 32 bit input
// value "val" when executed.
void CreatePrintfInt(Module *mod, BasicBlock *bb, Value *val) {
  assert(val && "invalid integer");
  Function *func_printf = __GetOrCreatePrintf(mod);

  IRBuilder<> builder(*TheContext);
  builder.SetInsertPoint(bb);
  Value *str = builder.CreateGlobalStringPtr("%d");

  std::vector<Value *> int32_call_params;
  int32_call_params.push_back(str);
  int32_call_params.push_back(val);

  CallInst::Create(func_printf, int32_call_params, "call", bb);
}

// ---------------------------------------------------------------------------
// Code generation functions that you should fill in for this assignment
// ---------------------------------------------------------------------------
Value *ProgramAST::codegen(Function *F) {
  // STUDENTS: FILL IN THIS FUNCTION
  for (auto &Stmt : Stmts) {
    Stmt->codegen(F);
  }
  return nullptr;
}

Value *AssignStmtAST::codegen(Function *F) {
  // STUDENTS: FILL IN THIS FUNCTION
  Value *rhs = RHS->codegen(F);
  if (!rhs) {
    return nullptr;
  }

  IRBuilder<> b(&F->getEntryBlock(),F->getEntryBlock().begin());
  // WHY IS IT SO HARD TO FIND DOCS https://docs.hdoc.io/hdoc/llvm-project/rB252D65A795AFC3C.html
  Value *val = b.CreateAlloca(rhs->getType(), nullptr);
  NamedValues[GetName()] = val;

  Builder.CreateStore(rhs, val);
  return rhs;
}

void PrintNestedArray(Value* V, MiniAPLArrayType &type, int start_idx=0) {
  int NDims = type.dimension();
  CreatePrintfStr(TheModule.get(), Builder.GetInsertBlock(), "[");
  if (NDims == 0) {
    Value *element_ptr = Builder.CreateGEP(intTy(32), V, intConst(32, start_idx));
    Value *val = Builder.CreateLoad(intTy(32), element_ptr);
    CreatePrintfInt(TheModule.get(), Builder.GetInsertBlock(), val);
  }
  /*else if (NDims == 1) {
    for (int i = 0; i < type.length(0); i++) {
      Value *element_ptr = Builder.CreateGEP(intTy(32), V, intConst(32, i + start_idx));
      Value *val = Builder.CreateLoad(intTy(32), element_ptr);
      CreatePrintfStr(TheModule.get(), Builder.GetInsertBlock(), "[");
      CreatePrintfInt(TheModule.get(), Builder.GetInsertBlock(), val);
      CreatePrintfStr(TheModule.get(), Builder.GetInsertBlock(), "]");
    }
  }*/
  else {
    std::vector<int> new_dims(type.dimensions.begin() + 1, type.dimensions.end());
    MiniAPLArrayType new_type = MiniAPLArrayType{new_dims};
    int n = type.Cardinality();
    int dim = type.length(0);
    for (int i = 0; i < dim; i++) {
      PrintNestedArray(V, new_type, start_idx + i*(n/dim));
    }
  }
  CreatePrintfStr(TheModule.get(), Builder.GetInsertBlock(), "]");
}


Value *ExprStmtAST::codegen(Function *F) {
  // STUDENTS: FILL IN THIS FUNCTION
  Value *val = Val->codegen(F);
  MiniAPLArrayType type = TypeTable[Val.get()];
  PrintNestedArray(val, type);
  CreatePrintfStr(TheModule.get(), Builder.GetInsertBlock(), "\n");
  return val;
}

Value *NumberASTNode::codegen(Function *F) {
  // STUDENTS: FILL IN THIS FUNCTION
  return intConst(32, Val);
}

Value *VariableASTNode::codegen(Function *F) {
  // STUDENTS: FILL IN THIS FUNCTION
  Value* var = NamedValues[Name];
  if (!var) {
    return nullptr;
  }
  return Builder.CreateLoad(var->getType(), var);
}

Value *CallASTNode::codegen(Function *F) {
  // STUDENTS: FILL IN THIS FUNCTION
  if (Callee == "mkArray") {
    int NDims = static_cast<NumberASTNode *>(Args.at(0).get())->Val;
    std::vector<int> Dims;
    for (int i = 0; i < NDims; i++) {
      Dims.push_back(static_cast<NumberASTNode *>(Args.at(i + 1).get())->Val);
    }
    MiniAPLArrayType type = MiniAPLArrayType{Dims};
    int n = type.Cardinality();

    // think this needs to be in entry cuz we're allocating
    IRBuilder<> b(&F->getEntryBlock(), F->getEntryBlock().begin());
    Value *alloca = b.CreateAlloca(intTy(32), intConst(32, n));

    int starting_elt_index = NDims + 1;
    for (int i = 0; i < n; i++) {
      Value *element_ptr = b.CreateGEP(intTy(32), alloca, intConst(32, i));
      int val = static_cast<NumberASTNode*>(Args.at(starting_elt_index +  i).get())->Val;
      b.CreateStore(intConst(32, val), element_ptr);
    }

    TypeTable[static_cast<ASTNode *>(this)] = type;
    return alloca;
  }
  else if (Callee == "add" || Callee == "sub") {
    ASTNode *a = Args.at(0).get();
    ASTNode *b = Args.at(1).get();
    if (!a || !b) {
      return nullptr;
    }
    MiniAPLArrayType a_type = TypeTable[a];
    MiniAPLArrayType b_type = TypeTable[b];
    assert(a_type.dimension() == b_type.dimension());
    for (int i = 0; i < a_type.dimension(); i++) {
      assert(a_type.dimensions[i] == b_type.dimensions[i]);
    }

    IRBuilder<> alloca_builder(&F->getEntryBlock(), F->getEntryBlock().begin());
    int n = a_type.Cardinality();
    Value *alloca = alloca_builder.CreateAlloca(intTy(32), intConst(32, n));
    Value* a_vals = a->codegen(F);
    Value* b_vals = b->codegen(F);
    for (int i = 0; i < n; i++) {
      Value *element_ptr = Builder.CreateGEP(intTy(32), alloca, intConst(32, i));
      Value *a_ptr = Builder.CreateGEP(intTy(32), a_vals, intConst(32, i));
      Value *b_ptr = Builder.CreateGEP(intTy(32), b_vals, intConst(32, i));
      Value *a_val = Builder.CreateLoad(intTy(32), a_ptr);
      Value *b_val = Builder.CreateLoad(intTy(32), b_ptr);
      Value *res;
      if (Callee == "add") {
        res = Builder.CreateAdd(a_val, b_val);
      } else {
        res = Builder.CreateSub(a_val, b_val);
      }
      Builder.CreateStore(res, element_ptr);
    }

    TypeTable[static_cast<ASTNode *>(this)] = a_type;
    return alloca;
  }
  else if (Callee == "reduce"){
    ASTNode* x =  Args.at(0).get();
    if (!x) return nullptr;
    MiniAPLArrayType type = TypeTable[x];
    assert(type.dimension() > 0);
    std::vector<int> new_dims(type.dimensions.begin() + 1, type.dimensions.end()); 
    MiniAPLArrayType new_type = MiniAPLArrayType{new_dims};

    IRBuilder<> alloca_builder(&F->getEntryBlock(), F->getEntryBlock().begin());
    Value *alloca = alloca_builder.CreateAlloca(intTy(32), intConst(32, new_type.Cardinality()));

    Value* vals = x->codegen(F);
    std::vector<Value*> new_vals;
    Value* computing = intConst(32, 0);
    int n_allocated = 0;
    for (int i = 0; i < type.Cardinality(); i++) {
      Value *ptr = Builder.CreateGEP(intTy(32), vals, intConst(32, i));
      Value* val = Builder.CreateLoad(intTy(32), ptr);
      computing = Builder.CreateAdd(computing, val);
      if ((i+1) % type.dimensions[type.dimension()-1] == 0) {
        Value *element_ptr = Builder.CreateGEP(intTy(32), alloca, intConst(32, n_allocated));
        n_allocated ++;
        Builder.CreateStore(computing, element_ptr);
        computing = intConst(32, 0);
      }
    }

    TypeTable[static_cast<ASTNode *>(this)] = new_type;
    return alloca;
  }
  else if (Callee == "exp") {
    ASTNode* x =  Args.at(0).get();
    int power = static_cast<NumberASTNode *>(Args.at(1).get())->Val;
    if (!x  || !power) return nullptr;
    MiniAPLArrayType type = TypeTable[x];

    int n = type.Cardinality();
    IRBuilder<> alloca_builder(&F->getEntryBlock(), F->getEntryBlock().begin());
    Value *alloca = alloca_builder.CreateAlloca(intTy(32), intConst(32, n));

    Value* vals = x->codegen(F);
    for (int i = 0; i < n; i++) {
      Value *element_ptr = Builder.CreateGEP(intTy(32), alloca, intConst(32, i)); 
      Value *ptr = Builder.CreateGEP(intTy(32), vals, intConst(32, i));
      Value* val = Builder.CreateLoad(intTy(32), ptr);
      Value *res = intConst(32, 1);
      for (int j = 0; j < power; j++) {
        res = Builder.CreateMul(res, val);
      }
      Builder.CreateStore(res, element_ptr);
    }

    TypeTable[static_cast<ASTNode *>(this)] = type;
    return alloca;
  }
  else if (Callee == "neg") {
    ASTNode* x =  Args.at(0).get();
    if (!x ) return nullptr;
    MiniAPLArrayType type = TypeTable[x];

    int n = type.Cardinality();
    IRBuilder<> alloca_builder(&F->getEntryBlock(), F->getEntryBlock().begin());
    Value *alloca = alloca_builder.CreateAlloca(intTy(32), intConst(32, n));

    Value* vals = x->codegen(F);
    for (int i = 0; i < n; i++) {
      Value *element_ptr = Builder.CreateGEP(intTy(32), alloca, intConst(32, i)); 
      Value *ptr = Builder.CreateGEP(intTy(32), vals, intConst(32, i));
      Value* val = Builder.CreateLoad(intTy(32), ptr);
      Value *res = Builder.CreateMul(intConst(32, -1), val);
      Builder.CreateStore(res, element_ptr);
    }

    TypeTable[static_cast<ASTNode *>(this)] = type;
    return alloca;
  }
  else if (Callee == "expand") {
    ASTNode* x =  Args.at(0).get();
    int size = static_cast<NumberASTNode *>(Args.at(1).get())->Val;
    if (!x  || !size) return nullptr;
    MiniAPLArrayType type = TypeTable[x];
    
    int n = type.Cardinality();
    IRBuilder<> alloca_builder(&F->getEntryBlock(), F->getEntryBlock().begin());
    Value *alloca = alloca_builder.CreateAlloca(intTy(32), intConst(32, size*n));

    Value* vals = x->codegen(F);
    int new_idx = 0;
    for (int i = 0; i < n; i++) {
      Value *ptr = Builder.CreateGEP(intTy(32), vals, intConst(32, i));
      Value* val = Builder.CreateLoad(intTy(32), ptr);
      for (int j = 0; j < size; j++) {
        Value *element_ptr = Builder.CreateGEP(intTy(32), alloca, intConst(32, i*size+j));
        Builder.CreateStore(val, element_ptr);
      }
    }

    std::vector<int> new_dims = type.dimensions;
    new_dims.push_back(size); 

    TypeTable[static_cast<ASTNode *>(this)] = MiniAPLArrayType{new_dims};
    return alloca;
  }
  else if (Callee == "concat") {
    ASTNode* a =  Args.at(0).get();
    ASTNode* b =  Args.at(1).get();
    int concat_dim = static_cast<NumberASTNode *>(Args.at(2).get())->Val;
    if (!a || !b) return nullptr;
    MiniAPLArrayType a_type = TypeTable[a];
    MiniAPLArrayType b_type = TypeTable[b];
    
    std::vector<int> dims;
    for (int i = 0; i < a_type.dimension(); i++) {
      if (i == concat_dim) {
        dims.push_back(a_type.dimensions[i]+b_type.dimensions[i]);
      }
      else {
        dims.push_back(a_type.dimensions[i]);
      }
    }
    MiniAPLArrayType new_type = MiniAPLArrayType{dims};

    int n = new_type.Cardinality();
    IRBuilder<> alloca_builder(&F->getEntryBlock(), F->getEntryBlock().begin());
    Value *alloca = alloca_builder.CreateAlloca(intTy(32), intConst(32, n));

    Value* a_vals = a->codegen(F);
    Value* b_vals = b->codegen(F);
    int a_idx = 0;
    int b_idx = 0;
    int do_a = 1;
    int do_b = 1;
    for (int i = new_type.dimension()-1; i >= concat_dim; i--) {
      do_a *= a_type.dimensions[i];
      do_b *= b_type.dimensions[i];
    }
    int switch_every = do_a + do_b;
    for (int i = 0; i < n; i++) {
      Value *element_ptr = Builder.CreateGEP(intTy(32), alloca, intConst(32, i));
      Value *ptr;
      if ((i % switch_every) < do_a) {
        ptr = Builder.CreateGEP(intTy(32), a_vals, intConst(32, a_idx));
        a_idx += 1;
      } else {
        ptr = Builder.CreateGEP(intTy(32), b_vals, intConst(32, b_idx));
        b_idx += 1;
      }
      Value* val = Builder.CreateLoad(intTy(32), ptr);
      Builder.CreateStore(val, element_ptr);
    }

    TypeTable[static_cast<ASTNode *>(this)] = new_type;
    return alloca;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Parser utilities
// ---------------------------------------------------------------------------
class ParseState {
public:
  int Position;
  std::vector<std::string> Tokens;

  ParseState(std::vector<std::string> &Tokens_)
      : Position(0), Tokens(Tokens_) {}

  bool AtEnd() { return Position == Tokens.size(); }

  std::string peek() {
    if (AtEnd()) {
      return "";
    }
    return Tokens.at(Position);
  }

  std::string peek(const int Offset) {
    assert(Position + Offset < Tokens.size());
    return Tokens.at(Position + Offset);
  }

  std::string eat() {
    std::string Current = peek();
    Position++;
    return Current;
  }
};

std::ostream &operator<<(std::ostream &out, ParseState &PS) {
  int i = 0;
  for (auto T : PS.Tokens) {
    if (i == PS.Position) {
      out << " | ";
    }
    out << T << " ";
    i++;
  }
  return out;
}

// If the next token is not `t`, log an error and return false.
// Otherwise return true.
bool EatOrError(ParseState &PS, std::string_view t) {
  if (PS.eat() == (t))
    return true;
  LogError("parsing error, expected: " + std::string(t));
  return false;
}

std::unique_ptr<ASTNode> ParseExpr(ParseState &PS) {
  std::string Name = PS.eat();
  if (is_int(Name)) {
    return std::unique_ptr<ASTNode>(new NumberASTNode(std::stoi(Name)));
  }

  bool isFunctionCall = PS.peek() == "(";
  if (!isFunctionCall)
    return std::unique_ptr<ASTNode>(new VariableASTNode(Name));

  PS.eat(); // consume "("

  std::vector<std::unique_ptr<ASTNode>> Args;
  while (PS.peek() != ")") {
    Args.push_back(ParseExpr(PS));
    if (PS.peek() != ")") {
      if (!EatOrError(PS, ","))
        return nullptr;
    }
  }
  if (!EatOrError(PS, ")"))
    return nullptr;

  return std::unique_ptr<ASTNode>(new CallASTNode(Name, std::move(Args)));
}

// ---------------------------------------------------------------------------
// Driver function for type-checking
// ---------------------------------------------------------------------------
void SetType(std::map<ASTNode *, MiniAPLArrayType> &Types, ASTNode *Expr) {
  if (Expr->GetType() == EXPR_TYPE_FUNCALL) {
    CallASTNode *Call = static_cast<CallASTNode *>(Expr);
    for (auto &A : Call->Args) {
      SetType(Types, A.get());
    }

    if (Call->Callee == "mkArray") {
      int NDims = static_cast<NumberASTNode *>(Call->Args.at(0).get())->Val;
      std::vector<int> Dims;
      for (int i = 0; i < NDims; i++) {
        Dims.push_back(
            static_cast<NumberASTNode *>(Call->Args.at(i + 1).get())->Val);
      }
      Types[Expr] = {Dims};
    } else if (Call->Callee == "reduce") {
      Types[Expr] = Types[Call->Args.back().get()];
      Types[Expr].dimensions.pop_back();
    } else if (Call->Callee == "add" || Call->Callee == "sub") {
      Types[Expr] = Types[Call->Args.at(0).get()];
    } else if (Call->Callee == "expand") {
      Types[Expr] = Types[Call->Args.at(0).get()];
      Types[Expr].dimensions.push_back(static_cast<NumberASTNode *>(Call->Args.at(1).get())->Val);
    } else if (Call->Callee == "concat") {
      std::vector<int> dims1 = Types[Call->Args.at(0).get()].dimensions;
      std::vector<int> dims2 = Types[Call->Args.at(1).get()].dimensions;
      int concat_dim = static_cast<NumberASTNode *>(Call->Args.at(2).get())->Val;
      assert(dims1.size() == dims2.size());
      int size = dims1.size();
      std::vector<int> dims;
      for (int i = 0; i < size; i++) {
        if (i == concat_dim) {
          dims.push_back(dims1[i]+dims2[i]);
        }
        else {
          assert(dims1[i] == dims2[i]);
          dims.push_back(dims1[i]);
        }
      }
      Types[Expr] = {dims};
    } else {
      Types[Expr] = Types[Call->Args.at(0).get()];
    }
  } else if (Expr->GetType() == EXPR_TYPE_SCALAR) {
    Types[Expr] = {{1}};
  } else if (Expr->GetType() == EXPR_TYPE_VARIABLE) {
    std::string ExprName = static_cast<VariableASTNode *>(Expr)->Name;
    for (auto T : Types) {
      auto V = T.first;
      if (V->GetType() == EXPR_TYPE_VARIABLE) {
        std::string Name = static_cast<VariableASTNode *>(V)->Name;
        if (Name == ExprName) {
          Types[Expr] = T.second;
        }
      }
    }
  }
}

// Run a set of optimization passes on this module.
void RunPasses(Module &M, MiniAPLJIT &JIT) {
  // Set backend for optimization passes and code generation.
  M.setTargetTriple(JIT.getTargetTriple().getTriple());

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(OptimizationLevel::O1);
  MPM.run(M, MAM);
}

int main(const int argc, char *argv[]) {
  assert(argc > 1);
  std::string file = argv[1];

  bool debug = false;
  if (argc > 2) {
    std::string D = argv[2];
    debug = (D == "-d" || D == "--debug");
  }

  std::ifstream t(file);
  std::string S((std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>());

  // Tokenize the file
  std::vector<std::string> Tokens;
  std::string NextToken = "";
  for (int i = 0; i < S.size(); i++) {
    char NC = S[i];
    if (NC == ',' || NC == '(' || NC == ')' || NC == ';' || NC == '=') {
      if (NextToken != "") {
        Tokens.push_back(NextToken);
      }
      NextToken = std::string("") + NC;
      Tokens.push_back(NextToken);
      NextToken = "";
    } else if (!isspace(NC)) {
      NextToken += NC;
    } else {
      assert(isspace(NC));
      if (NextToken != "") {
        Tokens.push_back(NextToken);
      }
      NextToken = "";
    }
  }
  if (NextToken != "") {
    Tokens.push_back(NextToken);
  }

  std::vector<std::vector<std::string>> Stmts;
  std::vector<std::string> Toks;
  for (std::string_view t : Tokens) {
    if (t == ";") {
      Stmts.push_back(Toks);
      Toks = {};
    } else {
      Toks.push_back(std::string(t));
    }
  }

  if (Toks.size() > 0) {
    Stmts.push_back(Toks);
  }

  // Parse each statement
  std::vector<std::unique_ptr<StmtAST>> ParsedStmts;
  for (std::vector<std::string> &S : Stmts) {
    ParseState PS(S);
    assert(S.size() > 0);
    if (PS.peek() != "assign") {
      std::unique_ptr<ASTNode> value = ParseExpr(PS);
      ParsedStmts.push_back(
          std::unique_ptr<StmtAST>(new ExprStmtAST(std::move(value))));
    } else {
      PS.eat(); // eat "assign"

      std::string Var = PS.eat();

      if (PS.eat() != "=") {
      } else {
        std::unique_ptr<ASTNode> value = ParseExpr(PS);
        ParsedStmts.push_back(
            std::unique_ptr<StmtAST>(new AssignStmtAST(Var, std::move(value))));
      }
    }
  }

  // Collect the statements into a program
  ProgramAST prog;
  prog.Stmts = std::move(ParsedStmts);

  // Infer types
  for (auto &S : prog.Stmts) {
    StmtAST *SA = S.get();
    if (SA->IsAssign()) {
      AssignStmtAST *Assign = static_cast<AssignStmtAST *>(SA);
      SetType(TypeTable, Assign->RHS.get());
      TypeTable[Assign->Name.get()] = TypeTable[Assign->RHS.get()];
    } else {
      ExprStmtAST *Expr = static_cast<ExprStmtAST *>(SA);
      SetType(TypeTable, Expr->Val.get());
    }
  }

  TheModule = std::make_unique<Module>("MiniAPL Module " + file, *TheContext);
  std::vector<Type *> Args(0, Type::getDoubleTy(*TheContext));
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(*TheContext), Args, false);

  static constexpr char entrySymbolName[] = "main";
  Function *F = Function::Create(FT, Function::ExternalLinkage, entrySymbolName,
                                 TheModule.get());
  BasicBlock::Create(*TheContext, "entry", F);
  Builder.SetInsertPoint(&(F->getEntryBlock()));

  prog.codegen(F);
  Builder.CreateRet(nullptr);

  if (debug) {
    TheModule->print(llvm::errs(), nullptr);
    llvm::errs() << "\n";
  }

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  // Create the JIT.
  Expected<MiniAPLJIT> TheJIT = MiniAPLJIT::Create();
  if (!TheJIT) {
    llvm::errs() << "unexpected JIT initialization failure: "
                 << TheJIT.takeError() << "\n";
    return 1;
  }

  if (debug && llvm::verifyModule(*TheModule, &llvm::errs())) {
    return 1;
  }

  // Run passes and update the Module with configurations from the JIT.
  RunPasses(*TheModule, *TheJIT);

  // Add the module to the JIT.
  ThreadSafeContext TSContext(std::move(TheContext));
  ThreadSafeModule TSM(std::move(TheModule), TSContext);
  if (Error E = TheJIT->addIRModule(std::move(TSM)); E) {
    llvm::errs() << "unexpected IR Module failure: " << E << "\n";
    return 1;
  }

  Expected mainFunc = TheJIT->lookup(entrySymbolName);
  if (!mainFunc) {
    llvm::errs() << "entry symbol not found: \"" << entrySymbolName
                 << "\", with error: " << mainFunc.takeError() << "\n";
    return 1;
  }

  assert(!mainFunc->isNull());
  auto *Main = mainFunc->toPtr<void (*)()>();
  Main();

  return 0;
}
